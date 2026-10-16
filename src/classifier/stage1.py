"""
Stage 1: Inexpensive Metadata & Coarse Feature Classifier
High-speed gate resolving clear video formats in sub-15ms.
"""

import os
import math
import numpy as np
import joblib
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.profiler import VideoMetadata, CoarseSignals


# Standard assignment taxonomy
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


class Stage1Features:
    """Extracts a fixed-length 17-dimensional vector from metadata and coarse signals."""

    FEATURE_NAMES = [
        "log_duration",
        "fps",
        "aspect_ratio",
        "log_resolution",
        "log_bitrate",
        "has_audio",
        "audio_channels",
        "file_size_mb",
        "hist_mean_r",
        "hist_mean_g",
        "hist_mean_b",
        "hist_std_r",
        "hist_std_g",
        "hist_std_b",
        "frame_diff_mean",
        "frame_diff_std",
        "cut_rate_estimate"
    ]

    @classmethod
    def transform(cls, meta: VideoMetadata, signals: CoarseSignals) -> np.ndarray:
        """Construct normalized feature vector."""
        duration = max(0.0, meta.duration_s)
        width = max(1, meta.width)
        height = max(1, meta.height)
        resolution = width * height
        bitrate = max(0.0, meta.bitrate_kbps)
        file_size_mb = meta.file_size_bytes / (1024.0 * 1024.0)

        vec = np.array([
            math.log1p(duration),
            meta.fps if meta.fps > 0 else 30.0,
            meta.aspect_ratio if meta.aspect_ratio > 0 else 1.778,
            math.log1p(resolution),
            math.log1p(bitrate),
            1.0 if meta.has_audio else 0.0,
            float(meta.audio_channels),
            math.log1p(file_size_mb),
            signals.hist_mean_r / 255.0,
            signals.hist_mean_g / 255.0,
            signals.hist_mean_b / 255.0,
            signals.hist_std_r / 128.0,
            signals.hist_std_g / 128.0,
            signals.hist_std_b / 128.0,
            signals.frame_diff_mean / 100.0,
            signals.frame_diff_std / 50.0,
            signals.cut_rate_estimate
        ], dtype=np.float32)

        # Replace any potential NaNs or Infs
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        return vec


class Stage1Classifier:
    """Stage 1 lightweight classifier model using standard scaled logistic regression."""

    def __init__(self, classes: Optional[List[str]] = None, random_state: int = 42):
        self.classes = classes or DEFAULT_CLASSES
        self.random_state = random_state
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

    def fit(self, X: np.ndarray, y: List[str]) -> "Stage1Classifier":
        """Fit model on feature matrix X and labels y."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distributions."""
        if not self.is_fitted:
            # Fallback uniform distribution if unfitted
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            n_classes = len(self.classes)
            return np.ones((n_samples, n_classes), dtype=np.float32) / float(n_classes)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def predict_single(self, feature_vec: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predict top class, confidence, and full probability map for one video."""
        probs = self.predict_proba(feature_vec)[0]
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
    def load(cls, filepath: str) -> "Stage1Classifier":
        """Load serialized model artifact from disk."""
        data = joblib.load(filepath)
        instance = cls(classes=data.get("classes"))
        instance.model = data["model"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance
