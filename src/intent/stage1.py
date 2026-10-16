"""
Stage 1: Fast Lexical Gate for Task 2 User Intent Classification.
Uses word/char n-gram TF-IDF + 34 grammatical cues + calibrated Logistic Regression.
Provides sub-millisecond CPU classification and confidence/margin metrics for early-exit decisions.
"""

import os
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from scipy.optimize import minimize

from src.intent.features import HybridFeaturePipeline, normalize_query, GrammaticalCueExtractor

INTENT_CLASSES = ["whole_video", "frame_level", "clip_level"]


class TemperatureScaler:
    """Platt / Temperature scaling for probability calibration."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(temperature, 0.05)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Optimize temperature parameter T to minimize negative log likelihood."""
        def nll_loss(t):
            temp = max(t[0], 0.05)
            scaled = logits / temp
            probs = softmax(scaled, axis=1)
            # Clip for numerical stability
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            # Compute cross entropy
            log_p = np.log(probs[np.arange(len(labels)), labels])
            return -np.mean(log_p)

        res = minimize(nll_loss, [self.temperature], bounds=[(0.05, 5.0)], method="L-BFGS-B")
        if res.success:
            self.temperature = float(res.x[0])
        return self

    def scale(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits / self.temperature, axis=1)


class Stage1IntentClassifier:
    """Stage 1 fast gate classifier combining TF-IDF, grammatical cues, and calibrated LR."""

    def __init__(
        self,
        c_param: float = 1.0,
        classes: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.classes = classes or INTENT_CLASSES
        self.c_param = c_param
        self.random_state = random_state
        self.feature_pipeline = HybridFeaturePipeline()
        self.scaler = StandardScaler(with_mean=False)
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )
        self.temperature_scaler = TemperatureScaler(1.0)
        self.is_fitted = False

    def fit(
        self,
        queries: List[str],
        labels: List[str],
        val_queries: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None
    ) -> "Stage1IntentClassifier":
        """Fit feature vectorizer, linear classifier, and calibration temperature."""
        X = self.feature_pipeline.fit_transform(queries)
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, labels)
        self.is_fitted = True

        # Fit temperature on validation set if provided, else on training logits
        if val_queries and val_labels:
            val_X = self.transform_features(val_queries)
            val_logits = self.classifier.decision_function(val_X)
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

    def transform_features(self, queries: List[str]) -> np.ndarray:
        """Extract and scale features for incoming queries."""
        X = self.feature_pipeline.transform(queries)
        return self.scaler.transform(X)

    def predict_proba(self, query: str) -> Dict[str, float]:
        """Compute calibrated class posterior probabilities for a single query."""
        if not self.is_fitted:
            raise RuntimeError("Stage1IntentClassifier is not fitted.")
        X = self.transform_features([query])
        logits = self.classifier.decision_function(X)
        if logits.ndim == 1:
            logits = np.vstack([-logits, logits]).T
        probs = self.temperature_scaler.scale(logits)[0]
        
        prob_dict = {}
        for cls_name, prob in zip(self.classifier.classes_, probs):
            prob_dict[cls_name] = float(prob)
        # Ensure all canonical classes are present
        for cls_name in self.classes:
            if cls_name not in prob_dict:
                prob_dict[cls_name] = 0.0
        return prob_dict

    def predict_with_confidence(self, query: str) -> Tuple[str, float, float, Dict[str, float]]:
        """Compute top intent class, top confidence, probability margin, and full posterior."""
        prob_dict = self.predict_proba(query)
        sorted_items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_cls, top_conf = sorted_items[0]
        second_conf = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = float(top_conf - second_conf)
        return top_cls, float(top_conf), margin, prob_dict

    def save(self, filepath: str) -> None:
        """Serialize fitted classifier and components to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        state = {
            "classes": self.classes,
            "c_param": self.c_param,
            "feature_pipeline": self.feature_pipeline,
            "scaler": self.scaler,
            "classifier": self.classifier,
            "temperature": self.temperature_scaler.temperature,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: str) -> "Stage1IntentClassifier":
        """Load serialized classifier from disk."""
        state = joblib.load(filepath)
        instance = cls(c_param=state["c_param"], classes=state["classes"])
        instance.feature_pipeline = state["feature_pipeline"]
        instance.scaler = state["scaler"]
        instance.classifier = state["classifier"]
        instance.temperature_scaler = TemperatureScaler(state["temperature"])
        instance.is_fitted = state["is_fitted"]
        return instance
