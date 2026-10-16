"""
Cascaded Hybrid Video Type Classifier
Combines fast metadata/coarse features (Stage 1) with adaptive frame CLIP embeddings (Stage 2).
Implements confidence-based routing, adaptive frame sampling, feature caching, and fallback handling.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from src.profiler import VideoMetadata, CoarseSignals, VideoProfiler
from src.classifier.stage1 import Stage1Classifier, Stage1Features, DEFAULT_CLASSES
from src.classifier.stage2 import Stage2Classifier, CLIPFeatureExtractor, AdaptiveFrameSampler

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_MAP = {
    "gameplay": "state_change_and_event",
    "egocentric": "activity_and_temporal",
    "podcast": "speech_turn_and_topic",
    "wildlife": "scene_and_activity",
    "sports": "event_and_play",
    "news_lecture": "topic_and_slide",
    "meetings": "speaker_turn_and_slide",
    "surveillance": "activity_and_time_window",
    "concert": "music_segment_and_performance",
    "film_animation": "scene_and_shot",
    "general": "adaptive_scene_and_semantic",
}


@dataclass
class CascadeResult:
    video_id: str
    predicted_class: str
    confidence: float
    stage_resolved: str  # "stage1", "stage2", or "fallback"
    escalated_to_stage2: bool
    stage1_class: Optional[str] = None
    stage1_confidence: Optional[float] = None
    stage2_class: Optional[str] = None
    stage2_confidence: Optional[float] = None
    chunking_strategy: str = "adaptive_scene_and_semantic"
    latency_ms: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "stage_resolved": self.stage_resolved,
            "escalated_to_stage2": self.escalated_to_stage2,
            "stage1_class": self.stage1_class,
            "stage1_confidence": round(self.stage1_confidence, 4) if self.stage1_confidence is not None else None,
            "stage2_class": self.stage2_class,
            "stage2_confidence": round(self.stage2_confidence, 4) if self.stage2_confidence is not None else None,
            "chunking_strategy": self.chunking_strategy,
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
        }


class FeatureCache:
    """Simple in-memory and disk-backed cache for extracted features and embeddings."""

    def __init__(self, cache_file: Optional[str] = None):
        self.cache_file = cache_file
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        if cache_file and os.path.exists(cache_file):
            self.load(cache_file)

    def _make_key(self, video_path: str) -> str:
        try:
            mtime = os.path.getmtime(video_path)
            size = os.path.getsize(video_path)
            base = os.path.basename(video_path)
            return f"{base}_{size}_{int(mtime)}"
        except Exception:
            return os.path.basename(video_path)

    def get(self, video_path: str) -> Optional[Dict[str, Any]]:
        key = self._make_key(video_path)
        return self._memory_cache.get(key)

    def set(self, video_path: str, data: Dict[str, Any]) -> None:
        key = self._make_key(video_path)
        self._memory_cache[key] = data

    def save(self, filepath: Optional[str] = None) -> None:
        target = filepath or self.cache_file
        if not target:
            return
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        import joblib
        joblib.dump(self._memory_cache, target)

    def load(self, filepath: str) -> None:
        import joblib
        if os.path.exists(filepath):
            try:
                self._memory_cache = joblib.load(filepath)
            except Exception as e:
                logger.warning(f"Could not load feature cache from {filepath}: {e}")


class CascadedClassifier:
    """
    Cascaded Hybrid Video Type Classifier.
    - Stage 1: Fast metadata and coarse signal features (<15ms).
    - Stage 2: Adaptive frame sampling + CLIP visual representations.
    - Router: Routes to Stage 2 if Stage 1 confidence < escalation_threshold.
    - Fallback: Defaults to 'general' if uncertain or corrupted.
    """

    def __init__(
        self,
        stage1: Optional[Stage1Classifier] = None,
        stage2: Optional[Stage2Classifier] = None,
        escalation_threshold: float = 0.85,
        fallback_threshold: float = 0.35,
        fallback_class: str = "general",
        strategy_map_path: Optional[str] = "data/video_type_mapping.json",
        profiler: Optional[VideoProfiler] = None,
        cache_path: Optional[str] = None,
    ):
        self.classes = DEFAULT_CLASSES
        self.stage1 = stage1 or Stage1Classifier(classes=self.classes)
        self.stage2 = stage2 or Stage2Classifier(classes=self.classes)
        self.escalation_threshold = escalation_threshold
        self.fallback_threshold = fallback_threshold
        self.fallback_class = fallback_class
        self.profiler = profiler or VideoProfiler()
        self.strategy_map = self._load_strategy_map(strategy_map_path)
        self.cache = FeatureCache(cache_path) if cache_path else None

    def _load_strategy_map(self, path: Optional[str]) -> Dict[str, str]:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("routing_strategy_map", DEFAULT_STRATEGY_MAP)
            except Exception as e:
                logger.warning(f"Failed to load strategy map from {path}: {e}")
        return DEFAULT_STRATEGY_MAP

    def predict_video(
        self,
        video_path: str,
        meta: Optional[VideoMetadata] = None,
        frames: Optional[List[np.ndarray]] = None,
        use_cache: bool = True,
    ) -> CascadeResult:
        """Run cascaded prediction pipeline on a single video file."""
        t0 = time.perf_counter()
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 0. Check feature cache if enabled
        cached = self.cache.get(video_path) if (use_cache and self.cache) else None
        cached_meta = cached.get("meta") if cached else None
        cached_s1_feat = cached.get("s1_feat") if cached else None
        cached_s2_emb = cached.get("s2_emb") if cached else None

        # 1. Metadata inspection
        if meta is None:
            meta = cached_meta or self.profiler.probe(video_path)

        if not meta.is_valid:
            latency = (time.perf_counter() - t0) * 1000.0
            return CascadeResult(
                video_id=video_id,
                predicted_class=self.fallback_class,
                confidence=0.0,
                stage_resolved="fallback",
                escalated_to_stage2=False,
                chunking_strategy=self.strategy_map.get(self.fallback_class, "adaptive_scene_and_semantic"),
                latency_ms=latency,
                error_message=meta.error_message or "Video could not be opened or has zero duration",
            )

        # 2. Stage 1 Feature Extraction
        if cached_s1_feat is not None:
            feat1 = cached_s1_feat
        else:
            if frames is None:
                frames = self.profiler.extract_frames(video_path, num_frames=4)
            signals = self.profiler.compute_coarse_signals(frames)
            feat1 = Stage1Features.transform(meta, signals)
            if use_cache and self.cache:
                self.cache.set(video_path, {"meta": meta, "s1_feat": feat1})

        # 3. Stage 1 Inference
        s1_class, s1_conf, s1_probs = self.stage1.predict_single(feat1)

        # 4. Confidence-based Routing Gate
        if s1_conf >= self.escalation_threshold:
            # Confident prediction at Stage 1
            latency = (time.perf_counter() - t0) * 1000.0
            strategy = self.strategy_map.get(s1_class, "adaptive_scene_and_semantic")
            return CascadeResult(
                video_id=video_id,
                predicted_class=s1_class,
                confidence=s1_conf,
                stage_resolved="stage1",
                escalated_to_stage2=False,
                stage1_class=s1_class,
                stage1_confidence=s1_conf,
                chunking_strategy=strategy,
                latency_ms=latency,
                probabilities=s1_probs,
            )

        # 5. Escalation to Stage 2 (Uncertain prediction)
        if cached_s2_emb is not None:
            s2_emb = cached_s2_emb
        else:
            n_frames = AdaptiveFrameSampler.get_target_frame_count(meta.duration_s)
            s2_frames = self.profiler.extract_frames(video_path, num_frames=n_frames)
            s2_emb = self.stage2.feature_extractor.extract_frame_embedding(s2_frames)
            if use_cache and self.cache:
                existing = self.cache.get(video_path) or {}
                existing["s2_emb"] = s2_emb
                self.cache.set(video_path, existing)

        s2_class, s2_conf, s2_probs = self.stage2.predict_single(s2_emb)

        # 6. Fallback Check for Unknown or Low Confidence
        if s2_conf < self.fallback_threshold:
            final_class = self.fallback_class
            stage_resolved = "fallback"
            confidence = s2_conf
        else:
            final_class = s2_class
            stage_resolved = "stage2"
            confidence = s2_conf

        latency = (time.perf_counter() - t0) * 1000.0
        strategy = self.strategy_map.get(final_class, "adaptive_scene_and_semantic")

        return CascadeResult(
            video_id=video_id,
            predicted_class=final_class,
            confidence=confidence,
            stage_resolved=stage_resolved,
            escalated_to_stage2=True,
            stage1_class=s1_class,
            stage1_confidence=s1_conf,
            stage2_class=s2_class,
            stage2_confidence=s2_conf,
            chunking_strategy=strategy,
            latency_ms=latency,
            probabilities=s2_probs,
        )

    def predict_batch(self, video_paths: List[str], use_cache: bool = True) -> List[CascadeResult]:
        """Process multiple videos sequentially with progress tracking."""
        results = []
        for path in video_paths:
            res = self.predict_video(path, use_cache=use_cache)
            results.append(res)
        return results

    def fit(
        self,
        s1_X: np.ndarray,
        s1_y: List[str],
        s2_X: np.ndarray,
        s2_y: List[str],
    ) -> "CascadedClassifier":
        """Fit both Stage 1 and Stage 2 models on respective feature matrices."""
        logger.info(f"Fitting Stage 1 on {len(s1_y)} samples...")
        self.stage1.fit(s1_X, s1_y)
        logger.info(f"Fitting Stage 2 on {len(s2_y)} samples...")
        self.stage2.fit(s2_X, s2_y)
        return self

    def save(self, model_dir: str) -> None:
        """Persist cascade model checkpoints and config."""
        os.makedirs(model_dir, exist_ok=True)
        self.stage1.save(os.path.join(model_dir, "stage1.joblib"))
        self.stage2.save(os.path.join(model_dir, "stage2.joblib"))
        config = {
            "escalation_threshold": self.escalation_threshold,
            "fallback_threshold": self.fallback_threshold,
            "fallback_class": self.fallback_class,
            "classes": self.classes,
        }
        with open(os.path.join(model_dir, "cascade_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        if self.cache:
            self.cache.save(os.path.join(model_dir, "feature_cache.joblib"))

    @classmethod
    def load(cls, model_dir: str, strategy_map_path: Optional[str] = "data/video_type_mapping.json") -> "CascadedClassifier":
        """Load cascade model and subcomponents from directory."""
        config_path = os.path.join(model_dir, "cascade_config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        stage1 = Stage1Classifier.load(os.path.join(model_dir, "stage1.joblib"))
        stage2 = Stage2Classifier.load(os.path.join(model_dir, "stage2.joblib"))

        cache_path = os.path.join(model_dir, "feature_cache.joblib")
        instance = cls(
            stage1=stage1,
            stage2=stage2,
            escalation_threshold=config.get("escalation_threshold", 0.85),
            fallback_threshold=config.get("fallback_threshold", 0.35),
            fallback_class=config.get("fallback_class", "general"),
            strategy_map_path=strategy_map_path,
            cache_path=cache_path if os.path.exists(cache_path) else None,
        )
        return instance
