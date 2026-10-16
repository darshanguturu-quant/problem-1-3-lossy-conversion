"""
Baseline Models and Benchmark Suite for Task 2 User Intent Classification.
Compares:
1. TF-IDF-only Classifier (Stage 1 alone)
2. Sentence-Embedding-only Classifier (Stage 2 alone)
3. Rule-Based Heuristic Matcher
4. Cascaded Hybrid Classifier (TF-IDF Gate + Conditional Sentence Embeddings)

Measures: Macro-F1, per-class P/R/F1, Expected Calibration Error (ECE), Brier Score,
P50/P95 latency, peak RAM memory, on-disk model size, and escalation rate.
"""

import os
import time
import json
import logging
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.intent.features import normalize_query
from src.intent.stage1 import Stage1IntentClassifier, INTENT_CLASSES
from src.intent.stage2 import Stage2IntentClassifier
from src.intent.cascade import CascadedIntentClassifier, IntentCascadeResult

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across M equal-width bins."""
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(
    probabilities: np.ndarray,
    ground_truth_indices: np.ndarray,
    num_classes: int = 3
) -> float:
    """Compute multi-class Brier score."""
    if len(probabilities) == 0:
        return 0.0

    n = len(ground_truth_indices)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    for i, idx in enumerate(ground_truth_indices):
        if 0 <= idx < num_classes:
            one_hot[i, idx] = 1.0

    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


@dataclass
class IntentMetricReport:
    """Metrics container for an intent classification architecture."""
    model_name: str
    macro_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]
    expected_calibration_error: float
    brier_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    peak_memory_mb: float
    model_size_mb: float
    escalation_rate_pct: float
    catastrophic_error_rate_pct: float
    ambiguity_rate_pct: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "macro_f1": round(self.macro_f1, 4),
            "per_class_metrics": {
                c: {k: round(v, 4) for k, v in m.items()}
                for c, m in self.per_class_metrics.items()
            },
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "escalation_rate_pct": round(self.escalation_rate_pct, 2),
            "catastrophic_error_rate_pct": round(self.catastrophic_error_rate_pct, 2),
            "ambiguity_rate_pct": round(self.ambiguity_rate_pct, 2),
            "num_samples": self.num_samples
        }


class RuleBasedIntentBaseline:
    """Handcrafted regex baseline for lexical query routing."""

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or INTENT_CLASSES

    def predict(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        q = normalize_query(query)
        if any(w in q for w in ["summar", "entire", "whole video", "all in all", "overview", "main topic", "focus of the video"]):
            return "whole_video", 0.85, {"whole_video": 0.85, "frame_level": 0.05, "clip_level": 0.10}
        if any(w in q for w in ["text", "scoreboard", "what color", "wearing", "sign", "slide", "written", "logo"]):
            return "frame_level", 0.85, {"whole_video": 0.05, "frame_level": 0.85, "clip_level": 0.10}
        if any(w in q for w in ["when does", "how did", "moment", "action", "kick", "run", "segment", "clip", "show the"]):
            return "clip_level", 0.85, {"whole_video": 0.10, "frame_level": 0.10, "clip_level": 0.80}
        return "clip_level", 0.40, {"whole_video": 0.30, "frame_level": 0.30, "clip_level": 0.40}


class IntentBenchmarkRunner:
    """Executes empirical benchmarking across all intent routing alternatives."""

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or INTENT_CLASSES

    def evaluate_model(
        self,
        model_name: str,
        predict_fn: Callable[[str], Tuple[str, float, Dict[str, float], bool]],
        queries: List[str],
        ground_truth: List[str],
        model_size_mb: float = 0.0
    ) -> IntentMetricReport:
        """Measure latency, memory, accuracy, calibration, and escalation rate."""
        latencies = []
        predictions = []
        confidences = []
        prob_matrix = []
        escalated_flags = []
        ambiguous_count = 0

        tracemalloc.start()
        t_start = time.perf_counter()

        for q in queries:
            t0 = time.perf_counter()
            pred_cls, conf, probs, is_esc = predict_fn(q)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            predictions.append(pred_cls)
            confidences.append(conf)
            escalated_flags.append(is_esc)

            if pred_cls in ["ambiguous", "needs_review"]:
                ambiguous_count += 1

            p_row = [probs.get(c, 0.0) for c in self.classes]
            # Normalize row sum
            row_sum = sum(p_row)
            if row_sum > 0:
                p_row = [x / row_sum for x in p_row]
            else:
                p_row = [1.0 / len(self.classes)] * len(self.classes)
            prob_matrix.append(p_row)

        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latencies_arr = np.array(latencies)
        p50 = float(np.percentile(latencies_arr, 50))
        p95 = float(np.percentile(latencies_arr, 95))
        peak_mb = peak_mem / (1024.0 * 1024.0)
        esc_rate = (sum(escalated_flags) / max(len(queries), 1)) * 100.0
        amb_rate = (ambiguous_count / max(len(queries), 1)) * 100.0

        # Mapping ambiguous predictions to clip_level for clean metric computation
        eval_preds = [p if p in self.classes else "clip_level" for p in predictions]

        # Macro-F1 and per-class metrics
        macro_f1 = float(f1_score(ground_truth, eval_preds, average="macro", zero_division=0))
        p, r, f, s = precision_recall_fscore_support(ground_truth, eval_preds, labels=self.classes, zero_division=0)

        per_class = {}
        for idx, c in enumerate(self.classes):
            per_class[c] = {
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f[idx]),
                "support": int(s[idx])
            }

        # Calibration metrics
        accuracies = np.array([1.0 if pred == true else 0.0 for pred, true in zip(eval_preds, ground_truth)])
        conf_arr = np.array(confidences)
        ece = compute_expected_calibration_error(conf_arr, accuracies)

        gt_indices = np.array([self.classes.index(lbl) if lbl in self.classes else -1 for lbl in ground_truth])
        prob_arr = np.array(prob_matrix, dtype=np.float32)
        brier = compute_brier_score(prob_arr, gt_indices, num_classes=len(self.classes))

        # Catastrophic error rate: whole_video true, but frame_level predicted
        wv_total = sum(1 for y in ground_truth if y == "whole_video")
        catastrophic_errors = sum(1 for y, p in zip(ground_truth, eval_preds) if y == "whole_video" and p == "frame_level")
        catastrophic_rate = (catastrophic_errors / max(wv_total, 1)) * 100.0

        return IntentMetricReport(
            model_name=model_name,
            macro_f1=macro_f1,
            per_class_metrics=per_class,
            expected_calibration_error=ece,
            brier_score=brier,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            peak_memory_mb=peak_mb,
            model_size_mb=model_size_mb,
            escalation_rate_pct=esc_rate,
            catastrophic_error_rate_pct=catastrophic_rate,
            ambiguity_rate_pct=amb_rate,
            num_samples=len(queries)
        )

    def run_benchmark(
        self,
        stage1_model: Stage1IntentClassifier,
        stage2_model: Stage2IntentClassifier,
        cascade_model: CascadedIntentClassifier,
        test_queries: List[str],
        ground_truth: List[str],
        save_dir: Optional[str] = "artifacts"
    ) -> Dict[str, IntentMetricReport]:
        """Run full evaluation comparing Baseline 1 (TF-IDF), Baseline 2 (Embeddings), Rule-based, and Cascade."""
        reports = {}

        # 1. Baseline 1: Rule-Based Heuristic
        rule_model = RuleBasedIntentBaseline(self.classes)
        def predict_rule(q: str):
            cls_name, conf, probs = rule_model.predict(q)
            return cls_name, conf, probs, False

        reports["rule_based"] = self.evaluate_model(
            "Rule-Based Regex Baseline",
            predict_rule,
            test_queries,
            ground_truth,
            model_size_mb=0.001
        )

        # 2. Baseline 2: TF-IDF + Logistic Regression (Stage 1 Alone)
        tmp_s1 = "artifacts/tmp_stage1.joblib"
        stage1_model.save(tmp_s1)
        s1_size = os.path.getsize(tmp_s1) / (1024.0 * 1024.0)
        if os.path.exists(tmp_s1):
            os.remove(tmp_s1)

        def predict_tfidf(q: str):
            c_name, conf, _margin, probs = stage1_model.predict_with_confidence(q)
            return c_name, conf, probs, False

        reports["tfidf_only"] = self.evaluate_model(
            "TF-IDF + LR (Stage 1 Alone)",
            predict_tfidf,
            test_queries,
            ground_truth,
            model_size_mb=s1_size
        )

        # 3. Baseline 3: Sentence-Embedding Classifier (Stage 2 Alone)
        tmp_s2 = "artifacts/tmp_stage2.joblib"
        stage2_model.save(tmp_s2)
        # Add underlying backbone footprint (~88 MB for MiniLM or 350 MB for CLIP)
        backbone_mb = 88.0 if stage2_model.extractor.backend == "sentence_transformers" else 350.0
        s2_size = (os.path.getsize(tmp_s2) / (1024.0 * 1024.0)) + backbone_mb
        if os.path.exists(tmp_s2):
            os.remove(tmp_s2)

        def predict_embedding(q: str):
            raw_probs, _ = stage2_model.predict_proba(q)
            top_c = max(raw_probs, key=raw_probs.get)
            return top_c, raw_probs[top_c], raw_probs, True

        reports["embedding_only"] = self.evaluate_model(
            "Sentence-Embedding Only (Stage 2 Alone)",
            predict_embedding,
            test_queries,
            ground_truth,
            model_size_mb=s2_size
        )

        # 4. Cascaded Hybrid Router
        cascade_size = s1_size + s2_size
        def predict_cascade(q: str):
            res = cascade_model.predict(q)
            return res.predicted_intent, res.confidence, res.probabilities, res.escalated_to_stage2

        reports["cascaded_hybrid"] = self.evaluate_model(
            "Cascaded Hybrid (TF-IDF Gate + Sentence Embeddings)",
            predict_cascade,
            test_queries,
            ground_truth,
            model_size_mb=cascade_size
        )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, "intent_benchmark_results.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in reports.items()}, f, indent=2)

        return reports

    @staticmethod
    def format_markdown_table(reports: Dict[str, IntentMetricReport]) -> str:
        """Render markdown comparison table."""
        header = (
            "| Model Architecture | Macro-F1 | ECE | P50 Latency | P95 Latency | Peak RAM | Model Size | Escalation Rate | Catastrophic Regret |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for key, r in reports.items():
            row = (
                f"| **{r.model_name}** | {r.macro_f1:.4f} | {r.expected_calibration_error:.4f} | "
                f"{r.p50_latency_ms:.2f} ms | {r.p95_latency_ms:.2f} ms | "
                f"{r.peak_memory_mb:.1f} MB | {r.model_size_mb:.1f} MB | "
                f"{r.escalation_rate_pct:.1f}% | {r.catastrophic_error_rate_pct:.1f}% |"
            )
            rows.append(row)
        return header + "\n".join(rows)
