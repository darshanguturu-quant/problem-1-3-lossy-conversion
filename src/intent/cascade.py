"""
Cascaded Hybrid Intent Router for Task 2 User Intent Classification.
Coordinates fast Stage 1 gate (TF-IDF + LR) and conditional Stage 2 semantic refinement (sentence embeddings).
Supports ambiguity handling returning 'ambiguous' or 'needs_review' when confidence is insufficient.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Literal
import numpy as np

from src.intent.stage1 import Stage1IntentClassifier, INTENT_CLASSES
from src.intent.stage2 import Stage2IntentClassifier

logger = logging.getLogger(__name__)

AmbiguityMode = Literal["ambiguous", "needs_review"]


@dataclass
class IntentCascadeResult:
    """Standardized output container for intent routing decisions."""
    query: str
    predicted_intent: str  # "whole_video", "frame_level", "clip_level", "ambiguous", or "needs_review"
    confidence: float
    stage_resolved: str  # "stage1", "stage2", or "ambiguous_fallback"
    escalated_to_stage2: bool
    stage1_class: Optional[str] = None
    stage1_confidence: Optional[float] = None
    stage1_margin: Optional[float] = None
    stage2_class: Optional[str] = None
    stage2_confidence: Optional[float] = None
    latency_ms: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    routing_target: str = "clip_index"
    utility_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "predicted_intent": self.predicted_intent,
            "confidence": round(self.confidence, 4),
            "stage_resolved": self.stage_resolved,
            "escalated_to_stage2": self.escalated_to_stage2,
            "stage1_class": self.stage1_class,
            "stage1_confidence": round(self.stage1_confidence, 4) if self.stage1_confidence is not None else None,
            "stage1_margin": round(self.stage1_margin, 4) if self.stage1_margin is not None else None,
            "stage2_class": self.stage2_class,
            "stage2_confidence": round(self.stage2_confidence, 4) if self.stage2_confidence is not None else None,
            "latency_ms": round(self.latency_ms, 3),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "routing_target": self.routing_target,
            "utility_score": round(self.utility_score, 4)
        }


class CascadedIntentClassifier:
    """Two-stage hybrid router coordinating Stage 1 fast gate and Stage 2 dense semantic refinement."""

    # Default routing targets
    TARGET_MAP = {
        "whole_video": "summary_index",
        "frame_level": "frame_index",
        "clip_level": "clip_index",
        "ambiguous": "human_review",
        "needs_review": "human_review"
    }

    # Accuracy / information preservation matrix A(true, routed)
    # [whole_video, frame_level, clip_level]
    ACCURACY_MATRIX = np.array([
        [1.00, 0.00, 0.60],  # true = whole_video: frame is catastrophic loss (0.0)!
        [0.40, 1.00, 0.75],  # true = frame_level: clip retains frame (0.75)
        [0.50, 0.20, 1.00],  # true = clip_level: frame loses temporal context (0.20)
    ], dtype=np.float32)

    COMPUTE_COSTS = np.array([10.0, 1.0, 3.0], dtype=np.float32)  # [whole_video, frame, clip]

    def __init__(
        self,
        stage1_classifier: Stage1IntentClassifier,
        stage2_classifier: Stage2IntentClassifier,
        escalation_threshold: float = 0.82,
        margin_threshold: float = 0.30,
        ambiguity_threshold: float = 0.45,
        ambiguity_mode: AmbiguityMode = "ambiguous",
        cost_lambda: float = 0.05
    ):
        self.stage1 = stage1_classifier
        self.stage2 = stage2_classifier
        self.escalation_threshold = escalation_threshold
        self.margin_threshold = margin_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.ambiguity_mode = ambiguity_mode
        self.cost_lambda = cost_lambda
        self.classes = stage1_classifier.classes

    def _compute_bayes_optimal_class(self, posteriors: Dict[str, float]) -> Tuple[str, float]:
        """Compute intent class maximizing expected retrieval utility."""
        p_vec = np.array([posteriors.get(c, 0.0) for c in self.classes], dtype=np.float32)
        norm_costs = self.COMPUTE_COSTS / np.max(self.COMPUTE_COSTS)

        expected_utilities = []
        for r_idx in range(len(self.classes)):
            # Sum over true intents t: p(t|q) * (A(t, r) - lambda * C(r))
            util = np.sum(p_vec * (self.ACCURACY_MATRIX[:, r_idx] - self.cost_lambda * norm_costs[r_idx]))
            expected_utilities.append(util)

        best_idx = int(np.argmax(expected_utilities))
        return self.classes[best_idx], float(expected_utilities[best_idx])

    def predict(self, query: str) -> IntentCascadeResult:
        """Route incoming query through the two-stage hybrid cascade."""
        t0 = time.perf_counter()

        # --- Stage 1: Fast Lexical Gate ---
        s1_class, s1_conf, s1_margin, s1_probs = self.stage1.predict_with_confidence(query)

        # Early exit check
        if s1_conf >= self.escalation_threshold and s1_margin >= self.margin_threshold:
            bayes_target_cls, util_score = self._compute_bayes_optimal_class(s1_probs)
            latency = (time.perf_counter() - t0) * 1000.0
            return IntentCascadeResult(
                query=query,
                predicted_intent=s1_class,
                confidence=s1_conf,
                stage_resolved="stage1",
                escalated_to_stage2=False,
                stage1_class=s1_class,
                stage1_confidence=s1_conf,
                stage1_margin=s1_margin,
                latency_ms=latency,
                probabilities=s1_probs,
                routing_target=self.TARGET_MAP.get(bayes_target_cls, "clip_index"),
                utility_score=util_score
            )

        # --- Stage 2: High-Capacity Semantic Refinement ---
        raw_s2_probs, blended_probs = self.stage2.predict_proba(query, stage1_prob=s1_probs)
        sorted_s2 = sorted(blended_probs.items(), key=lambda kv: kv[1], reverse=True)
        s2_top_cls, s2_top_conf = sorted_s2[0]

        # Ambiguity handling: when confidence remains insufficient
        if s2_top_conf < self.ambiguity_threshold:
            latency = (time.perf_counter() - t0) * 1000.0
            return IntentCascadeResult(
                query=query,
                predicted_intent=self.ambiguity_mode,
                confidence=s2_top_conf,
                stage_resolved="ambiguous_fallback",
                escalated_to_stage2=True,
                stage1_class=s1_class,
                stage1_confidence=s1_conf,
                stage1_margin=s1_margin,
                stage2_class=s2_top_cls,
                stage2_confidence=s2_top_conf,
                latency_ms=latency,
                probabilities=blended_probs,
                routing_target=self.TARGET_MAP.get(self.ambiguity_mode, "human_review"),
                utility_score=0.5
            )

        # Bayes-optimal target selection on blended posterior
        bayes_target_cls, util_score = self._compute_bayes_optimal_class(blended_probs)
        latency = (time.perf_counter() - t0) * 1000.0
        return IntentCascadeResult(
            query=query,
            predicted_intent=s2_top_cls,
            confidence=s2_top_conf,
            stage_resolved="stage2",
            escalated_to_stage2=True,
            stage1_class=s1_class,
            stage1_confidence=s1_conf,
            stage1_margin=s1_margin,
            stage2_class=s2_top_cls,
            stage2_confidence=s2_top_conf,
            latency_ms=latency,
            probabilities=blended_probs,
            routing_target=self.TARGET_MAP.get(bayes_target_cls, "clip_index"),
            utility_score=util_score
        )

    def tune_thresholds(
        self,
        val_queries: List[str],
        val_labels: List[str],
        target_escalation_min: float = 0.10,
        target_escalation_max: float = 0.25
    ) -> Dict[str, float]:
        """Tune escalation and margin thresholds strictly on validation data."""
        best_f1 = -1.0
        best_params = {
            "escalation_threshold": self.escalation_threshold,
            "margin_threshold": self.margin_threshold,
            "ambiguity_threshold": self.ambiguity_threshold
        }

        # Grid of candidate thresholds
        cand_escalate = [0.75, 0.80, 0.82, 0.85, 0.88]
        cand_margin = [0.20, 0.25, 0.30, 0.35]

        from sklearn.metrics import f1_score

        for th in cand_escalate:
            for mg in cand_margin:
                self.escalation_threshold = th
                self.margin_threshold = mg

                preds = []
                escalated_count = 0
                for q in val_queries:
                    res = self.predict(q)
                    preds.append(res.predicted_intent)
                    if res.escalated_to_stage2:
                        escalated_count += 1

                esc_rate = escalated_count / max(len(val_queries), 1)

                # Map ambiguous predictions to most frequent class for F1 calculation
                clean_preds = [p if p in self.classes else "clip_level" for p in preds]
                score = f1_score(val_labels, clean_preds, average="macro", zero_division=0)

                # Prioritize parameter sets within target escalation window
                if target_escalation_min <= esc_rate <= target_escalation_max:
                    if score > best_f1:
                        best_f1 = score
                        best_params = {"escalation_threshold": th, "margin_threshold": mg, "ambiguity_threshold": self.ambiguity_threshold}

        # Apply best found parameters
        self.escalation_threshold = best_params["escalation_threshold"]
        self.margin_threshold = best_params["margin_threshold"]
        logger.info("Tuned thresholds on validation: %s (Val Macro-F1=%.4f)", best_params, best_f1)
        return best_params
