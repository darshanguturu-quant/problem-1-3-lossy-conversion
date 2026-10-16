"""
Codec Benchmarking Suite (Task 3)
Benchmarks candidate codecs (H.264, H.265, AV1, and OpenCV fallback) on a sample video.
Records output size, compression ratio, encode/decode latency, quality measurements,
and empirical ROI metrics to produce an evidence-backed architectural scorecard.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2

from src.transcoder.engine import TranscodeEngine, TranscodeResult
from src.transcoder.profiles import CodecProfile, get_profile, adapt_profile_for_content
from src.transcoder.profiler import VideoProbeResult, VideoProfilerFFprobe
from src.transcoder.quality import QualityReport, measure_video_quality
from src.transcoder.roi import ConversionDecisionEngine, CostModelParameters, ROIEstimate

logger = logging.getLogger("aintropy-transcoder.benchmark")


@dataclass
class CodecBenchmarkEntry:
    """Benchmark results for a single codec profile."""
    profile_name: str
    video_codec: str
    backend: str
    success: bool
    original_size_mb: float
    output_size_mb: float
    size_reduction_pct: float
    compression_ratio: float
    encode_time_s: float
    encode_speed_factor: float
    decode_speed_fps: float
    psnr_db: float
    ssim: float
    edge_preservation: float
    net_financial_savings_usd: float
    net_time_savings_s: float
    break_even_passes: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodecBenchmarkScorecard:
    """Aggregated multi-codec benchmark scorecard with comparative ranking."""
    video_path: str
    duration_s: float
    original_size_mb: float
    original_codec: str
    entries: List[CodecBenchmarkEntry] = field(default_factory=list)
    recommended_profile: Optional[str] = None
    selection_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration_s": self.duration_s,
            "original_size_mb": self.original_size_mb,
            "original_codec": self.original_codec,
            "recommended_profile": self.recommended_profile,
            "selection_rationale": self.selection_rationale,
            "entries": [e.to_dict() for e in self.entries]
        }

    def format_markdown_table(self) -> str:
        """Render side-by-side comparative Markdown table."""
        lines = [
            "| Codec Profile | Backend | Out Size (MB) | Ratio | Encode Speed | PSNR (dB) | SSIM | Net ROI ($) | Net Time (s) |",
            "|---|---|---|---|---|---|---|---|---|"
        ]
        for e in self.entries:
            if not e.success:
                lines.append(f"| **{e.profile_name}** | {e.backend} | *FAILED* | - | - | - | - | - | {e.error_message or 'Error'} |")
                continue
            ratio_str = f"{e.compression_ratio:.2f}"
            speed_str = f"{e.encode_speed_factor:.1f}x"
            psnr_str = f"{e.psnr_db:.1f}"
            ssim_str = f"{e.ssim:.3f}"
            roi_str = f"${e.net_financial_savings_usd:.4f}"
            time_str = f"{e.net_time_savings_s:+.1f}s"
            lines.append(
                f"| **{e.profile_name}** | `{e.backend}` | {e.output_size_mb:.2f} MB | {ratio_str} | {speed_str} | {psnr_str} | {ssim_str} | {roi_str} | {time_str} |"
            )
        return "\n".join(lines)


class CodecBenchmarkRunner:
    """Executes multi-codec benchmarking across configured profiles."""

    def __init__(
        self,
        output_dir: str = "artifacts/derived",
        cost_params: Optional[CostModelParameters] = None
    ):
        self.output_dir = output_dir
        self.cost_params = cost_params or CostModelParameters()
        self.profiler = VideoProfilerFFprobe()
        self.engine = TranscodeEngine(default_output_dir=output_dir)
        self.decision_engine = ConversionDecisionEngine(self.cost_params)

    def benchmark(
        self,
        video_path: str,
        profile_names: Optional[List[str]] = None,
        content_genre: Optional[str] = None
    ) -> CodecBenchmarkScorecard:
        """Run benchmark on video across specified codec profiles."""
        orig_meta = self.profiler.probe(video_path)
        if not orig_meta.is_valid:
            scorecard = CodecBenchmarkScorecard(
                video_path=video_path,
                duration_s=0.0,
                original_size_mb=0.0,
                original_codec="unknown",
                recommended_profile=None,
                selection_rationale=f"Failed to probe video: {orig_meta.error_message}"
            )
            return scorecard

        # Default profile candidate selection
        candidates = profile_names
        if not candidates:
            # If FFmpeg is available, benchmark H.264 baseline and H.265 storage
            if self.engine.is_ffmpeg_available:
                candidates = ["h264_baseline", "h265_storage"]
            else:
                candidates = ["opencv_fallback"]

        entries: List[CodecBenchmarkEntry] = []
        os.makedirs(self.output_dir, exist_ok=True)

        for pname in candidates:
            base_profile = get_profile(pname)
            adapted = adapt_profile_for_content(base_profile, orig_meta, content_genre=content_genre)

            out_path = self.engine.generate_output_path(video_path, adapted, output_dir=self.output_dir)
            t_res = self.engine.transcode(video_path, adapted, output_path=out_path, duration_s=orig_meta.duration_s)

            if not t_res.success:
                entries.append(
                    CodecBenchmarkEntry(
                        profile_name=adapted.name,
                        video_codec=adapted.video_codec,
                        backend=t_res.backend,
                        success=False,
                        original_size_mb=orig_meta.file_size_mb,
                        output_size_mb=0.0,
                        size_reduction_pct=0.0,
                        compression_ratio=1.0,
                        encode_time_s=t_res.elapsed_time_s,
                        encode_speed_factor=0.0,
                        decode_speed_fps=0.0,
                        psnr_db=0.0,
                        ssim=0.0,
                        edge_preservation=0.0,
                        net_financial_savings_usd=0.0,
                        net_time_savings_s=0.0,
                        break_even_passes=float("inf"),
                        error_message=t_res.error_message
                    )
                )
                continue

            trans_meta = self.profiler.probe(t_res.output_path)
            q_res = measure_video_quality(video_path, t_res.output_path, num_eval_frames=5)
            roi = self.decision_engine.evaluate_measured(
                orig_meta, trans_meta, t_res.elapsed_time_s, params=self.cost_params
            )

            # Measure decode playback speed (FPS)
            decode_fps = self._measure_decode_speed(t_res.output_path)

            size_reduc_pct = round((1.0 - t_res.compression_ratio) * 100.0, 1) if t_res.compression_ratio < 1.0 else 0.0

            entries.append(
                CodecBenchmarkEntry(
                    profile_name=adapted.name,
                    video_codec=adapted.video_codec,
                    backend=t_res.backend,
                    success=True,
                    original_size_mb=orig_meta.file_size_mb,
                    output_size_mb=trans_meta.file_size_mb,
                    size_reduction_pct=size_reduc_pct,
                    compression_ratio=t_res.compression_ratio,
                    encode_time_s=t_res.elapsed_time_s,
                    encode_speed_factor=t_res.speed_factor,
                    decode_speed_fps=decode_fps,
                    psnr_db=q_res.psnr_db,
                    ssim=q_res.ssim,
                    edge_preservation=q_res.edge_preservation_ratio,
                    net_financial_savings_usd=roi.net_financial_savings_usd,
                    net_time_savings_s=roi.net_time_savings_s,
                    break_even_passes=roi.break_even_passes
                )
            )

        # Select recommended profile based on empirical evidence:
        # Priority: (meets quality constraints) -> (highest net financial savings)
        valid_entries = [e for e in entries if e.success and e.ssim >= 0.85 and e.psnr_db >= 30.0]
        if valid_entries:
            # Sort by net financial savings descending
            valid_entries.sort(key=lambda x: x.net_financial_savings_usd, reverse=True)
            best = valid_entries[0]
            recommended = best.profile_name
            rationale = (
                f"Selected '{best.profile_name}' based on highest net financial savings (${best.net_financial_savings_usd:.4f}) "
                f"while maintaining high visual quality (PSNR {best.psnr_db:.1f} dB, SSIM {best.ssim:.3f})."
            )
        elif entries and any(e.success for e in entries):
            # Fallback to the successful entry with lowest size
            succ = [e for e in entries if e.success]
            succ.sort(key=lambda x: x.output_size_mb)
            recommended = succ[0].profile_name
            rationale = f"Selected '{recommended}' as primary valid transcode candidate."
        else:
            recommended = None
            rationale = "No candidate codec completed successfully."

        return CodecBenchmarkScorecard(
            video_path=video_path,
            duration_s=orig_meta.duration_s,
            original_size_mb=orig_meta.file_size_mb,
            original_codec=orig_meta.codec,
            entries=entries,
            recommended_profile=recommended,
            selection_rationale=rationale
        )

    def _measure_decode_speed(self, video_path: str, max_frames: int = 30) -> float:
        """Measure sequential decode frames-per-second."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0

        start_t = time.perf_counter()
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            count += 1
        elapsed = max(0.001, time.perf_counter() - start_t)
        cap.release()
        return round(count / elapsed, 1)
