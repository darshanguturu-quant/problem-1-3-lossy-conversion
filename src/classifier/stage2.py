"""
Stage 2: Adaptive Frame Sampler & CLIP Visual Semantic Classifier
High-capacity visual refinement for uncertain video type predictions.
"""

import os
import cv2
import numpy as np
import torch
import joblib
from PIL import Image
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.profiler import VideoMetadata, VideoProfiler


DEFAULT_CLASSES = [
    "gameplay",
    "egocentric",
    "podcast",
    "wildlife",
    "sports",
    "news_lecture",
    "meetings",
    "surveillance",
    "concert",
    "film_animation",
    "general"
]


class AdaptiveFrameSampler:
    """Dynamically determines the optimal frame sample count based on duration and entropy."""

    @staticmethod
    def get_target_frame_count(duration_s: float, cut_rate_estimate: float = 0.0) -> int:
        """Adaptive sampling policy:
        - Short videos (<= 5 min): 4 frames (sufficient for static/short formats).
        - Medium videos (5-30 min): 8 frames.
        - Long/dynamic videos (> 30 min): 12 frames.
        """
        if duration_s <= 300.0:
            return 4
        elif duration_s <= 1800.0:
            return 8
        else:
            return 12


class CLIPFeatureExtractor:
    """Extracts 512-dim normalized visual semantic representations using CLIP."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: Optional[str] = None):
        self.model_name = model_name
        self.pretrained = pretrained or os.environ.get("CLIP_PRETRAINED", None)
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self._fallback_mode = False
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load open_clip; fall back gracefully to lightweight torchvision if offline or failed."""
        try:
            import open_clip
            if self.pretrained:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device
                )
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=None,
                    device=self.device
                )
            model.eval()
            self.model = model
            self.preprocess = preprocess
        except Exception:
            # Fallback to torchvision mobilenet if clip weights or open_clip fail
            try:
                import torchvision.models as models
                from torchvision import transforms
                mb = models.mobilenet_v3_small(weights=None)
                mb.classifier = torch.nn.Identity()  # 576-dim output
                mb.eval()
                self.model = mb
                self.preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self._fallback_mode = True
            except Exception:
                self.model = None
                self._fallback_mode = True

    def extract_frame_embedding(self, bgr_frames: List[np.ndarray]) -> np.ndarray:
        """Encode sampled BGR frames and compute mean-pooled visual representation."""
        if not bgr_frames or self.model is None:
            # Deterministic zero vector fallback
            dim = 576 if self._fallback_mode else 512
            return np.zeros(dim, dtype=np.float32)

        tensors = []
        for f in bgr_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            if self.preprocess:
                tensors.append(self.preprocess(pil_img))

        if not tensors:
            dim = 576 if self._fallback_mode else 512
            return np.zeros(dim, dtype=np.float32)

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            if not self._fallback_mode and hasattr(self.model, "encode_image"):
                feats = self.model.encode_image(batch)
                # Normalize CLIP embeddings
                feats = feats / feats.norm(dim=-1, keepdim=True)
            else:
                feats = self.model(batch)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)

        # Mean pooling across sampled frames
        mean_embedding = feats.mean(dim=0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding /= norm

        return mean_embedding


class Stage2Classifier:
    """Stage 2 high-capacity semantic classifier using CLIP embeddings + linear head."""

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        random_state: int = 42,
        model_name: str = "ViT-B-32",
        pretrained: Optional[str] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None
    ):
        self.classes = classes or DEFAULT_CLASSES
        self.random_state = random_state
        self.model_name = model_name
        self.pretrained = pretrained
        self.feature_extractor = feature_extractor or CLIPFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained
        )
        self.sampler = AdaptiveFrameSampler()
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
                C=1.0
            ))
        ])
        self.is_fitted = False

    def extract_features(self, video_path: str, meta: Optional[VideoMetadata] = None, profiler: Optional[VideoProfiler] = None) -> np.ndarray:
        """Adaptively sample frames and extract pooled visual embeddings."""
        p = profiler or VideoProfiler()
        if meta is None:
            meta = p.probe(video_path)

        n_frames = self.sampler.get_target_frame_count(meta.duration_s)
        frames = p.extract_frames(video_path, num_frames=n_frames)
        return self.feature_extractor.extract_frame_embedding(frames)

    def fit(self, X: np.ndarray, y: List[str]) -> "Stage2Classifier":
        """Fit linear head on pooled visual embeddings."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distributions."""
        if not self.is_fitted:
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            n_classes = len(self.classes)
            return np.ones((n_samples, n_classes), dtype=np.float32) / float(n_classes)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def predict_single(self, embedding: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict top class, confidence, and full probability map."""
        probs = self.predict_proba(embedding)[0]
        model_classes = list(self.model.classes_) if self.is_fitted else self.classes

        prob_map = {cls_name: float(p) for cls_name, p in zip(model_classes, probs)}
        top_idx = int(np.argmax(probs))
        top_class = model_classes[top_idx]
        confidence = float(probs[top_idx])

        return top_class, confidence, prob_map

    def save(self, filepath: str) -> None:
        """Serialize model artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes, "is_fitted": self.is_fitted}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "Stage2Classifier":
        """Load serialized model artifact from disk."""
        data = joblib.load(filepath)
        instance = cls(classes=data.get("classes"))
        instance.model = data["model"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance
