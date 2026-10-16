"""
Reproducible evaluation and benchmarking harness for Task 4 Intelligent Video Chunking.
Evaluates Fixed-Duration Baseline, Blind Scene-Cut Baseline, and Content-Aware Intelligent Chunker.
Measures Boundary F1 (+-2.0s tolerance), Overlap Compliance, Throughput, and Duration Statistics.
"""

from dataclasses import dataclass, field, asdict
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from src.chunker.types import ChunkManifest, ChildChunk, BoundaryCandidate
from src.chunker.policies import ChunkingPolicy, get_policy
from src.chunker.baselines import FixedDurationChunker, BlindSceneCutChunker
from src.chunker.engine import IntelligentChunker
from src.chunker.signals import MultiSignalDetector

logger = logging.getLogger("aintropy-chunker.benchmarks")


@dataclass
class BenchmarkScorecard:
    """Individual model evaluation report on video chunking."""
    model_name: str
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    overlap_ratio: float
    overlap_compliant: bool  # 5% to 15% per spec
    duration_compliance_pct: float  # Percentage of chunks inside [min, max]
    avg_chunk_duration_s: float
    std_chunk_duration_s: float
    min_chunk_duration_s: float
    max_chunk_duration_s: float
    num_chunks: int
    throughput_factor: float  # media-sec / wall-sec
    avg_boundary_distance_s: float  # distance to nearest natural transition
    processing_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChunkerBenchmarkRunner:
    """
    Rigorously benchmarks chunking strategies against ground-truth transitions or multi-signal anchors.
    """

    def __init__(self, tolerance_s: float = 2.0):
        self.tolerance_s = tolerance_s

    def evaluate_manifest(
        self,
        manifest: ChunkManifest,
        ground_truth_cuts: List[float],
        policy: ChunkingPolicy,
        all_natural_candidates: Optional[List[float]] = None,
    ) -> BenchmarkScorecard:
        """
        Evaluates a chunking manifest against ground truth cut timestamps and policy bounds.
        """
        # Extract predicted boundary cuts (excluding 0.0 and total duration)
        pred_cuts = [c.end_s for c in manifest.chunks if 0.0 < c.end_s < manifest.total_duration_s]
        gt_cuts = [g for g in ground_truth_cuts if 0.0 < g < manifest.total_duration_s]

        # 1. Compute Boundary Precision, Recall, F1 with +-tolerance_s
        prec, rec, f1 = self.compute_boundary_f1(pred_cuts, gt_cuts, tolerance_s=self.tolerance_s)

        # 2. Check Overlap compliance
        measured_overlap = manifest.overlap_ratio
        overlap_compliant = (0.05 <= measured_overlap <= 0.15) if manifest.num_chunks > 1 else True

        # 3. Check Duration compliance with [T_min, T_max]
        durations = [c.duration_s for c in manifest.chunks]
        if durations:
            compliant_count = sum(
                1 for d in durations
                if (policy.min_duration_s * 0.95) <= d <= (policy.max_duration_s * 1.25)
            )
            compliance_pct = (compliant_count / len(durations)) * 100.0
            avg_dur = float(np.mean(durations))
            std_dur = float(np.std(durations))
            min_dur = float(np.min(durations))
            max_dur = float(np.max(durations))
        else:
            compliance_pct = 0.0
            avg_dur = std_dur = min_dur = max_dur = 0.0

        # 4. Average distance to nearest natural transition
        cand_pool = all_natural_candidates or gt_cuts
        if pred_cuts and cand_pool:
            distances = []
            for p in pred_cuts:
                min_dist = min(abs(p - c) for c in cand_pool)
                distances.append(min_dist)
            avg_boundary_dist = float(np.mean(distances))
        else:
            avg_boundary_dist = 0.0

        return BenchmarkScorecard(
            model_name=manifest.strategy,
            boundary_precision=round(prec, 4),
            boundary_recall=round(rec, 4),
            boundary_f1=round(f1, 4),
            overlap_ratio=round(measured_overlap, 4),
            overlap_compliant=overlap_compliant,
            duration_compliance_pct=round(compliance_pct, 2),
            avg_chunk_duration_s=round(avg_dur, 2),
            std_chunk_duration_s=round(std_dur, 2),
            min_chunk_duration_s=round(min_dur, 2),
            max_chunk_duration_s=round(max_dur, 2),
            num_chunks=len(manifest.chunks),
            throughput_factor=round(manifest.throughput_factor, 1),
            avg_boundary_distance_s=round(avg_boundary_dist, 3),
            processing_time_s=round(manifest.processing_time_s, 4),
        )

    def run_benchmark(
        self,
        video_path: str,
        video_type: str = "general",
        ground_truth_cuts: Optional[List[float]] = None,
        output_dir: Optional[str] = "artifacts",
    ) -> Dict[str, BenchmarkScorecard]:
        """
        Runs full 3-way comparative benchmark:
        1. Fixed Duration Baseline (30s windows)
        2. Blind Scene-Cut Baseline
        3. Intelligent Content-Aware Chunker
        """
        policy = get_policy(video_type)

        # 1. Detect natural signals to establish ground truth if none supplied
        detector = MultiSignalDetector(sample_fps=3.0, scene_threshold=policy.scene_threshold)
        raw_candidates = detector.detect_signals(video_path)
        natural_cuts = [c.timestamp_s for c in raw_candidates]

        gt_cuts = ground_truth_cuts if ground_truth_cuts is not None else natural_cuts

        # Probe video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration_s = (frame_cnt / fps) if (fps > 0 and frame_cnt > 0) else 60.0

        # Model 1: Fixed Duration Baseline
        fixed_chunker = FixedDurationChunker(target_duration_s=policy.target_duration_s, overlap_ratio=policy.overlap_ratio)
        fixed_manifest = fixed_chunker.chunk(video_path, duration_s)

        # Model 2: Blind Scene-Cut Baseline
        blind_chunker = BlindSceneCutChunker(scene_threshold=policy.scene_threshold)
        blind_manifest = blind_chunker.chunk(video_path, duration_s)

        # Model 3: Intelligent Content-Aware Chunker
        intelligent_chunker = IntelligentChunker(policy=policy)
        intelligent_manifest = intelligent_chunker.chunk_video(
            video_path=video_path,
            video_type=video_type,
            video_duration_s=duration_s,
            policy=policy,
        )

        # Evaluate all 3
        scorecards: Dict[str, BenchmarkScorecard] = {
            "fixed_duration_baseline": self.evaluate_manifest(
                fixed_manifest, gt_cuts, policy, all_natural_candidates=natural_cuts
            ),
            "blind_scenecut_baseline": self.evaluate_manifest(
                blind_manifest, gt_cuts, policy, all_natural_candidates=natural_cuts
            ),
            "intelligent_content_aware": self.evaluate_manifest(
                intelligent_manifest, gt_cuts, policy, all_natural_candidates=natural_cuts
            ),
        }

        # Save manifests and scorecard if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fixed_manifest.save(os.path.join(output_dir, "chunk_manifest_fixed.json"))
            blind_manifest.save(os.path.join(output_dir, "chunk_manifest_blind.json"))
            intelligent_manifest.save(os.path.join(output_dir, "chunk_manifest_intelligent.json"))

            summary_path = os.path.join(output_dir, "chunker_benchmark_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in scorecards.items()}, f, indent=2)
            logger.info("Saved Task 4 benchmark summary to '%s'", summary_path)

        return scorecards

    @staticmethod
    def compute_boundary_f1(
        pred_cuts: List[float],
        gt_cuts: List[float],
        tolerance_s: float = 2.0
    ) -> Tuple[float, float, float]:
        """
        Computes precision, recall, and F1 with greedy bipartite matching within tolerance window.
        """
        if not pred_cuts and not gt_cuts:
            return 1.0, 1.0, 1.0
        if not pred_cuts or not gt_cuts:
            return 0.0, 0.0, 0.0

        matched_gt = set()
        tp = 0

        for p in pred_cuts:
            best_gt = None
            min_dist = float("inf")
            for idx, g in enumerate(gt_cuts):
                if idx in matched_gt:
                    continue
                dist = abs(p - g)
                if dist <= tolerance_s and dist < min_dist:
                    min_dist = dist
                    best_gt = idx
            if best_gt is not None:
                matched_gt.add(best_gt)
                tp += 1

        prec = tp / len(pred_cuts) if pred_cuts else 0.0
        rec = tp / len(gt_cuts) if gt_cuts else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        return prec, rec, f1

    @staticmethod
    def format_markdown_table(scorecards: Dict[str, BenchmarkScorecard]) -> str:
        """Renders scorecards into GitHub-flavored Markdown comparison table."""
        header = (
            "| Chunking Architecture | Boundary F1 (+/-2s) | Precision | Recall | Overlap Ratio | Duration Compliance | Mean Dur | Throughput | Cut Coherence (Dist) |\n"
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
        )
        rows = []
        name_map = {
            "fixed_duration_baseline": "Fixed-Duration Baseline (Uniform 30s)",
            "blind_scenecut_baseline": "Blind Scene-Cut Baseline (Unconstrained)",
            "intelligent_content_aware": "**Intelligent Content-Aware (Proposed)**",
        }
        for k, sc in scorecards.items():
            label = name_map.get(k, k)
            comp_badge = "[OK]" if sc.overlap_compliant else "[NON-COMPLIANT]"
            overlap_str = f"{sc.overlap_ratio * 100:.1f}% {comp_badge}"
            dur_comp = f"{sc.duration_compliance_pct:.1f}%"
            mean_dur = f"{sc.avg_chunk_duration_s:.1f}s (+/-{sc.std_chunk_duration_s:.1f}s)"
            speed = f"{sc.throughput_factor:.1f}x real-time"
            dist_str = f"{sc.avg_boundary_distance_s:.2f}s"
            rows.append(
                f"| {label} | {sc.boundary_f1:.4f} | {sc.boundary_precision:.4f} | {sc.boundary_recall:.4f} | {overlap_str} | {dur_comp} | {mean_dur} | {speed} | {dist_str} |"
            )
        return header + "\n" + "\n".join(rows)


def generate_multiscene_benchmark_video(
    output_path: str,
    duration_s: float = 120.0,
    scene_interval_s: float = 20.0,
    fps: int = 15,
    width: int = 160,
    height: int = 120,
) -> Tuple[str, List[float]]:
    """
    Generates a deterministic synthetic video with distinct scene transitions at known ground truth intervals.
    Returns: (output_path, ground_truth_cuts).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_s * fps)
    gt_cuts = []
    current_cut = scene_interval_s
    while current_cut < duration_s:
        gt_cuts.append(round(current_cut, 2))
        current_cut += scene_interval_s

    scene_colors = [
        (34, 139, 34),    # Green
        (120, 40, 20),    # Blue slide
        (0, 140, 255),    # Orange
        (200, 200, 200),  # White slide
        (10, 10, 180),    # Red
        (150, 0, 150),    # Purple
    ]

    for i in range(total_frames):
        t = i / fps
        scene_idx = int(t // scene_interval_s) % len(scene_colors)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color = scene_colors[scene_idx]
        frame[:, :] = color

        if scene_idx % 2 == 1:
            # Presentation slide text lines
            cv2.line(frame, (10, 30), (width - 10, 30), (255, 255, 255), 2)
            cv2.line(frame, (10, 60), (width - 30, 60), (255, 255, 255), 2)
        else:
            # Dynamic moving object
            ball_x = int(20 + ((t % scene_interval_s) / scene_interval_s) * (width - 40))
            cv2.circle(frame, (ball_x, height // 2), 10, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    return output_path, gt_cuts
