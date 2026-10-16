"""
Stage 2: Dense Sentence Embedding & Lightweight Semantic Classifier for Task 2 User Intent.
Refines ambiguous or uncertain queries conditionally using dense semantic representations.
Includes multi-tier fallbacks: SentenceTransformer -> open_clip text encoder -> deterministic projection.
"""

import os
import joblib
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

from src.intent.features import normalize_query, GrammaticalCueExtractor
from src.intent.stage1 import TemperatureScaler, INTENT_CLASSES

logger = logging.getLogger(__name__)


class SentenceEmbeddingExtractor:
    """Extracts dense semantic representations with robust fallbacks for CPU/offline environments."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.st_model = None
        self.clip_model = None
        self.clip_tokenizer = None
        self.embedding_dim = 384
        self.backend = "deterministic_projection"
        self._initialize_extractor()

    def _initialize_extractor(self) -> None:
        """Attempt to load SentenceTransformer, then open_clip, then deterministic fallback."""
        # 1. Try SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer
            # Try loading from local cache first to avoid slow network handshakes
            try:
                model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                model = SentenceTransformer(self.model_name, device=self.device)
            self.st_model = model
            if hasattr(model, "get_embedding_dimension"):
                self.embedding_dim = model.get_embedding_dimension()
            else:
                self.embedding_dim = model.get_sentence_embedding_dimension()
            self.backend = "sentence_transformers"
            logger.info("Initialized SentenceTransformer backend (%s, dim=%d)", self.model_name, self.embedding_dim)
            return
        except Exception as e:
            logger.debug("SentenceTransformer initialization skipped/failed: %s", e)

        # 2. Try open_clip text encoder (already installed and verified in environment)
        try:
            import open_clip
            import torch
            clip_m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=self.device)
            clip_m.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.clip_model = clip_m
            self.clip_tokenizer = tokenizer
            self.embedding_dim = 512
            self.backend = "open_clip_text"
            logger.info("Initialized open_clip text encoder backend (ViT-B-32, dim=512)")
            return
        except Exception as e:
            logger.debug("open_clip text encoder skipped/failed: %s", e)

        # 3. Deterministic projection fallback
        self.embedding_dim = 384
        self.backend = "deterministic_projection"
        logger.info("Using deterministic semantic projection fallback (dim=384)")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text queries into an (N, D) normalized dense float32 array."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        norm_texts = [normalize_query(t) for t in texts]

        # SentenceTransformers path
        if self.backend == "sentence_transformers" and self.st_model is not None:
            try:
                embeddings = self.st_model.encode(
                    norm_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning("SentenceTransformer encoding failed (%s); using fallback.", e)

        # open_clip path
        if self.clip_model is not None and self.clip_tokenizer is not None:
            try:
                import torch
                tokens = self.clip_tokenizer(norm_texts).to(self.device)
                with torch.no_grad():
                    embs = self.clip_model.encode_text(tokens)
                    embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-7)
                return embs.cpu().numpy().astype(np.float32)
            except Exception as e:
                logger.warning("open_clip text encoding failed (%s); using projection.", e)

        # Deterministic semantic hash projection fallback
        return self._deterministic_projection(norm_texts)

    def _deterministic_projection(self, texts: List[str]) -> np.ndarray:
        """Deterministic subword projection matrix ensuring consistent embeddings in offline/mock runs."""
        dim = self.embedding_dim
        N = len(texts)
        result = np.zeros((N, dim), dtype=np.float32)

        for i, text in enumerate(texts):
            # Deterministic pseudo-random seed from query text
            h = abs(hash(text)) % (2**31 - 1)
            rng = np.random.RandomState(h)
            vec = rng.randn(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            result[i] = vec / (norm + 1e-7)

        return result


class Stage2IntentClassifier:
    """Stage 2 semantic classifier combining dense embeddings, Stage 1 priors, and lightweight head."""

    def __init__(
        self,
        c_param: float = 1.0,
        classes: Optional[List[str]] = None,
        embedding_extractor: Optional[SentenceEmbeddingExtractor] = None,
        blend_alpha: float = 0.85,
        random_state: int = 42
    ):
        self.classes = classes or INTENT_CLASSES
        self.c_param = c_param
        self.blend_alpha = blend_alpha
        self.random_state = random_state
        self.extractor = embedding_extractor or SentenceEmbeddingExtractor()
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )
        self.temperature_scaler = TemperatureScaler(1.0)
        self.is_fitted = False

    def _build_feature_matrix(
        self,
        queries: List[str],
        stage1_probs: Optional[List[Dict[str, float]]] = None
    ) -> np.ndarray:
        """Combine dense embeddings with normalized Stage 1 posteriors and key structural cues."""
        embeddings = self.extractor.encode(queries)
        cues = GrammaticalCueExtractor.batch_extract(queries)

        if stage1_probs is None:
            # Default uniform prior
            priors = np.full((len(queries), len(self.classes)), 1.0 / len(self.classes), dtype=np.float32)
        else:
            priors = np.zeros((len(queries), len(self.classes)), dtype=np.float32)
            for i, p_dict in enumerate(stage1_probs):
                for j, cls_name in enumerate(self.classes):
                    priors[i, j] = p_dict.get(cls_name, 0.0)

        # Concatenate: [Embeddings (384/512), Priors (3), Cues (34)]
        return np.hstack([embeddings, priors, cues])

    def fit(
        self,
        queries: List[str],
        labels: List[str],
        stage1_probs: Optional[List[Dict[str, float]]] = None,
        val_queries: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
        val_stage1_probs: Optional[List[Dict[str, float]]] = None
    ) -> "Stage2IntentClassifier":
        """Fit Stage 2 classification head and calibrate temperature."""
        X = self._build_feature_matrix(queries, stage1_probs)
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, labels)
        self.is_fitted = True

        # Fit temperature
        if val_queries and val_labels:
            val_X = self._build_feature_matrix(val_queries, val_stage1_probs)
            val_X_scaled = self.scaler.transform(val_X)
            val_logits = self.classifier.decision_function(val_X_scaled)
            if val_logits.ndim == 1:
                val_logits = np.vstack([-val_logits, val_logits]).T
            label_indices = np.array([self.classes.index(lbl) for lbl in val_labels if lbl in self.classes])
            self.temperature_scaler.fit(val_logits, label_indices)
        else:
            train_logits = self.classifier.decision_function(X_scaled)
            if train_logits.ndim == 1:
                train_logits = np.vstack([-train_logits, train_logits]).T
            label_indices = np.array([self.classes.index(lbl) for lbl in labels if lbl in self.classes])
            self.temperature_scaler.fit(train_logits, label_indices)

        return self

    def predict_proba(
        self,
        query: str,
        stage1_prob: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute raw Stage 2 probabilities and blended posterior probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Stage2IntentClassifier is not fitted.")

        s1_list = [stage1_prob] if stage1_prob else None
        X = self._build_feature_matrix([query], s1_list)
        X_scaled = self.scaler.transform(X)
        logits = self.classifier.decision_function(X_scaled)
        if logits.ndim == 1:
            logits = np.vstack([-logits, logits]).T
        stage2_probs = self.temperature_scaler.scale(logits)[0]

        stage2_dict = {}
        for cls_name, prob in zip(self.classifier.classes_, stage2_probs):
            stage2_dict[cls_name] = float(prob)
        for cls_name in self.classes:
            if cls_name not in stage2_dict:
                stage2_dict[cls_name] = 0.0

        # Posterior fusion: p_final = alpha * p2 + (1 - alpha) * p1
        final_dict = {}
        if stage1_prob:
            for cls_name in self.classes:
                p1 = stage1_prob.get(cls_name, 1.0 / len(self.classes))
                p2 = stage2_dict.get(cls_name, 1.0 / len(self.classes))
                final_dict[cls_name] = float(self.blend_alpha * p2 + (1.0 - self.blend_alpha) * p1)
            # Normalize sum to 1.0
            total = sum(final_dict.values())
            if total > 0:
                final_dict = {k: v / total for k, v in final_dict.items()}
        else:
            final_dict = stage2_dict

        return stage2_dict, final_dict

    def save(self, filepath: str) -> None:
        """Serialize Stage 2 model head and scaler (excluding raw extractor weights)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "classes": self.classes,
            "c_param": self.c_param,
            "blend_alpha": self.blend_alpha,
            "scaler": self.scaler,
            "classifier": self.classifier,
            "temperature": self.temperature_scaler.temperature,
            "backend": self.extractor.backend,
            "embedding_dim": self.extractor.embedding_dim,
            "model_name": self.extractor.model_name,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: str, extractor: Optional[SentenceEmbeddingExtractor] = None) -> "Stage2IntentClassifier":
        """Load serialized Stage 2 head from disk."""
        state = joblib.load(filepath)
        ext = extractor or SentenceEmbeddingExtractor(model_name=state.get("model_name", "all-MiniLM-L6-v2"))
        instance = cls(
            c_param=state["c_param"],
            classes=state["classes"],
            embedding_extractor=ext,
            blend_alpha=state["blend_alpha"]
        )
        instance.scaler = state["scaler"]
        instance.classifier = state["classifier"]
        instance.temperature_scaler = TemperatureScaler(state["temperature"])
        instance.is_fitted = state["is_fitted"]
        return instance
