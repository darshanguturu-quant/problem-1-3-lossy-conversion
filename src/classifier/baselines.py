"""
Baseline Models and Benchmark Suite for Task 1 Video Type Classification.
Compares:
1. Metadata-only Classifier (Stage 1 alone)
2. MobileNetV3-only Classifier (Lightweight CNN Baseline)
3. CLIP-only Classifier (Stage 2 alone)
4. Cascaded Hybrid Classifier (Stage 1 + Confidence Gate + Adaptive CLIP Stage 2)
"""

import os
import time
import json
import logging
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import joblib

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from src.profiler import VideoMetadata, CoarseSignals, VideoProfiler
from src.classifier.stage1 import Stage1Classifier, Stage1Features, DEFAULT_CLASSES
from src.classifier.stage2 import Stage2Classifier, CLIPFeatureExtractor, AdaptiveFrameSampler
from src.classifier.cascade import CascadedClassifier, CascadeResult

logger = logging.getLogger(__name__)


class MobileNetV3FeatureExtractor:
    """Extracts 576-dim pooled CNN visual features using MobileNetV3-Small."""

    def __init__(self):
        self.device = "cpu"
        mb = models.mobilenet_v3_small(weights=None)
        mb.classifier = torch.nn.Identity()  # 576-dim output
        mb.eval()
        self.model = mb
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frame_embedding(self, bgr_frames: List[np.ndarray]) -> np.ndarray:
        """Encode BGR frames and compute mean-pooled visual representation."""
        if not bgr_frames:
            return np.zeros(576, dtype=np.float32)

        import cv2
        tensors = []
        for f in bgr_frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensors.append(self.preprocess(pil_img))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)

        mean_embedding = feats.mean(dim=0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding /= norm
        return mean_embedding


class MobileNetV3Classifier:
    """MobileNetV3-Small baseline visual classifier with logistic regression head."""

    def __init__(self, classes: Optional[List[str]] = None, random_state: int = 42):
        self.classes = classes or DEFAULT_CLASSES
        self.random_state = random_state
        self.feature_extractor = MobileNetV3FeatureExtractor()
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
        p = profiler or VideoProfiler()
        if meta is None:
            meta = p.probe(video_path)
        n_frames = self.sampler.get_target_frame_count(meta.duration_s)
        frames = p.extract_frames(video_path, num_frames=n_frames)
        return self.feature_extractor.extract_frame_embedding(frames)

    def fit(self, X: np.ndarray, y: List[str]) -> "MobileNetV3Classifier":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
            n_classes = len(self.classes)
            return np.ones((n_samples, n_classes), dtype=np.float32) / float(n_classes)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def predict_single(self, embedding: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        probs = self.predict_proba(embedding)[0]
        model_classes = list(self.model.classes_) if self.is_fitted else self.classes
        prob_map = {cls_name: float(p) for cls_name, p in zip(model_classes, probs)}
        top_idx = int(np.argmax(probs))
        top_class = model_classes[top_idx]
        confidence = float(probs[top_idx])
        return top_class, confidence, prob_map

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes, "is_fitted": self.is_fitted}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "MobileNetV3Classifier":
        data = joblib.load(filepath)
        inst = cls(classes=data.get("classes"))
        inst.model = data["model"]
        inst.is_fitted = data.get("is_fitted", True)
        return inst


@dataclass
class BaselineMetricReport:
    model_name: str
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    peak_memory_mb: float
    model_size_mb: float
    escalation_rate_pct: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "macro_f1": round(self.macro_f1, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "escalation_rate_pct": round(self.escalation_rate_pct, 1),
            "sample_count": self.sample_count,
            "per_class_precision": {k: round(v, 4) for k, v in self.per_class_precision.items()},
            "per_class_recall": {k: round(v, 4) for k, v in self.per_class_recall.items()},
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
        }


class BenchmarkRunner:
    """Orchestrates comprehensive benchmark comparing all 4 models."""

    def __init__(self, profiler: Optional[VideoProfiler] = None):
        self.profiler = profiler or VideoProfiler()

    def evaluate_model(
        self,
        name: str,
        predict_fn,
        video_paths: List[str],
        ground_truth: List[str],
        model_size_mb: float = 0.0,
        is_cascade: bool = False,
    ) -> BaselineMetricReport:
        """Run latency, memory, and classification evaluation on a dataset split."""
        latencies: List[float] = []
        predictions: List[str] = []
        escalated_count = 0

        # Memory tracking
        tracemalloc.start()
        t_start = time.perf_counter()

        for path in video_paths:
            t0 = time.perf_counter()
            res = predict_fn(path)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            if isinstance(res, tuple):
                pred_cls, conf = res[0], res[1]
            elif isinstance(res, CascadeResult):
                pred_cls = res.predicted_class
                if res.escalated_to_stage2:
                    escalated_count += 1
            else:
                pred_cls = str(res)

            predictions.append(pred_cls)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak / (1024.0 * 1024.0)

        # Classification metrics
        macro_f1 = float(f1_score(ground_truth, predictions, average="macro", zero_division=0))
        report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)

        classes = sorted(list(set(ground_truth + predictions)))
        per_class_p = {}
        per_class_r = {}
        per_class_f = {}

        for cls_name in classes:
            if cls_name in report:
                per_class_p[cls_name] = float(report[cls_name]["precision"])
                per_class_r[cls_name] = float(report[cls_name]["recall"])
                per_class_f[cls_name] = float(report[cls_name]["f1-score"])

        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        if is_cascade:
            esc_rate = (escalated_count / len(video_paths) * 100.0) if video_paths else 0.0
        elif "CLIP" in name:
            esc_rate = 100.0
        else:
            esc_rate = 0.0

        return BaselineMetricReport(
            model_name=name,
            macro_f1=macro_f1,
            per_class_precision=per_class_p,
            per_class_recall=per_class_r,
            per_class_f1=per_class_f,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            mean_latency_ms=mean_lat,
            peak_memory_mb=peak_mb,
            model_size_mb=model_size_mb,
            escalation_rate_pct=esc_rate,
            sample_count=len(video_paths),
        )

    def run_all_baselines(
        self,
        video_paths: List[str],
        ground_truth: List[str],
        stage1_model: Stage1Classifier,
        mobilenet_model: MobileNetV3Classifier,
        stage2_model: Stage2Classifier,
        cascade_model: CascadedClassifier,
        save_dir: Optional[str] = "artifacts/task1_benchmark",
    ) -> Dict[str, BaselineMetricReport]:
        """Execute full benchmark across all 4 configurations."""
        reports: Dict[str, BaselineMetricReport] = {}

        # 1. Metadata-Only (Stage 1)
        def predict_meta(path):
            meta = self.profiler.probe(path)
            frames = self.profiler.extract_frames(path, num_frames=4)
            signals = self.profiler.compute_coarse_signals(frames)
            vec = Stage1Features.transform(meta, signals)
            return stage1_model.predict_single(vec)

        # Estimate Stage 1 model size
        tmp_s1 = "artifacts/tmp_s1.joblib"
        stage1_model.save(tmp_s1)
        s1_size = os.path.getsize(tmp_s1) / (1024.0 * 1024.0)
        if os.path.exists(tmp_s1):
            os.remove(tmp_s1)

        logger.info("Evaluating Baseline 1: Metadata-only...")
        reports["metadata_only"] = self.evaluate_model(
            "Metadata-only (Stage 1)",
            predict_meta,
            video_paths,
            ground_truth,
            model_size_mb=s1_size,
        )

        # 2. MobileNetV3-Only
        def predict_mobilenet(path):
            emb = mobilenet_model.extract_features(path, profiler=self.profiler)
            return mobilenet_model.predict_single(emb)

        tmp_mb = "artifacts/tmp_mb.joblib"
        mobilenet_model.save(tmp_mb)
        mb_size = (os.path.getsize(tmp_mb) / (1024.0 * 1024.0)) + 9.3  # include ~9.3MB backbone
        if os.path.exists(tmp_mb):
            os.remove(tmp_mb)

        logger.info("Evaluating Baseline 2: MobileNetV3-only...")
        reports["mobilenet_v3"] = self.evaluate_model(
            "MobileNetV3-only",
            predict_mobilenet,
            video_paths,
            ground_truth,
            model_size_mb=mb_size,
        )

        # 3. CLIP-Only (Stage 2)
        def predict_clip(path):
            emb = stage2_model.extract_features(path, profiler=self.profiler)
            return stage2_model.predict_single(emb)

        tmp_s2 = "artifacts/tmp_s2.joblib"
        stage2_model.save(tmp_s2)
        clip_backbone_mb = 350.0 if not stage2_model.feature_extractor._fallback_mode else 9.3
        s2_size = (os.path.getsize(tmp_s2) / (1024.0 * 1024.0)) + clip_backbone_mb
        if os.path.exists(tmp_s2):
            os.remove(tmp_s2)

        logger.info("Evaluating Baseline 3: CLIP-only (Stage 2)...")
        reports["clip_only"] = self.evaluate_model(
            "CLIP-only (Stage 2)",
            predict_clip,
            video_paths,
            ground_truth,
            model_size_mb=s2_size,
        )

        # 4. Cascaded Hybrid
        def predict_cascade(path):
            return cascade_model.predict_video(path, use_cache=False)

        cascade_size = s1_size + s2_size
        logger.info("Evaluating Baseline 4: Cascaded Hybrid Classifier...")
        reports["cascaded_hybrid"] = self.evaluate_model(
            "Cascaded Hybrid (Stage 1 + Stage 2)",
            predict_cascade,
            video_paths,
            ground_truth,
            model_size_mb=cascade_size,
            is_cascade=True,
        )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_file = os.path.join(save_dir, "benchmark_results.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in reports.items()}, f, indent=2)

        return reports

    @staticmethod
    def format_markdown_table(reports: Dict[str, BaselineMetricReport]) -> str:
        """Render beautiful markdown comparison table."""
        header = (
            "| Model Architecture | Macro-F1 | P50 Latency | P95 Latency | Peak RAM | Model Size | Escalation Rate |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for key, r in reports.items():
            row = (
                f"| **{r.model_name}** | {r.macro_f1:.4f} | "
                f"{r.p50_latency_ms:.1f} ms | {r.p95_latency_ms:.1f} ms | "
                f"{r.peak_memory_mb:.1f} MB | {r.model_size_mb:.1f} MB | "
                f"{r.escalation_rate_pct:.1f}% |"
            )
            rows.append(row)
        return header + "\n".join(rows)
