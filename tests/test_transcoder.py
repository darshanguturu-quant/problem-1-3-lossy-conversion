"""
Unit and Integration Tests for Task 3 — Lossy Conversion & ROI Engine.
Validates:
  1. Video profiling with FFprobe and OpenCV fallback.
  2. Break-even formula: Net savings = Original processing/I/O cost + storage savings - Compressed I/O - Encoding cost.
  3. Smart bypass logic for already-efficient bitrates (< 80 MB/min).
  4. Content-aware profile adaptation (sports, meetings/slides, 4K downscaling, 480p preservation).
  5. Objective quality evaluation (PSNR, SSIM, edge preservation).
  6. Non-destructive transcoding execution (original master unchanged).
  7. Multi-codec benchmarking and CLI integration.
"""

import json
import os
import subprocess
import sys
import numpy as np
import pytest

from src.transcoder.profiler import VideoProbeResult, VideoProfilerFFprobe
from src.transcoder.roi import (
    ConversionDecisionEngine,
    CostModelParameters,
    DecisionRecommendation,
    ROIEstimate,
)
from src.transcoder.profiles import (
    CodecProfile,
    get_profile,
    adapt_profile_for_content,
    AVAILABLE_PROFILES,
)
from src.transcoder.quality import (
    calculate_psnr,
    calculate_ssim,
    calculate_edge_preservation,
    measure_video_quality,
    QualityReport,
)
from src.transcoder.engine import TranscodeEngine, TranscodeResult
from src.transcoder.benchmark import CodecBenchmarkRunner, CodecBenchmarkScorecard


# =====================================================================
# 1. Video Profiling Tests
# =====================================================================

def test_profiler_valid_video(sample_sports_video):
    profiler = VideoProfilerFFprobe()
    res = profiler.probe(sample_sports_video)

    assert res.is_valid is True
    assert res.file_size_bytes > 0
    assert res.file_size_mb > 0.0
    assert res.duration_s > 0.0
    assert res.width == 320
    assert res.height == 240
    assert res.fps > 0.0
    assert res.codec in ["h264", "mpeg4", "unknown", "mp4v", "fmp4"]
    assert res.error_message is None


def test_profiler_nonexistent_file():
    profiler = VideoProfilerFFprobe()
    res = profiler.probe("path/to/definitely/nonexistent_video_99999.mp4")

    assert res.is_valid is False
    assert "File not found" in res.error_message


def test_profiler_corrupt_file(invalid_video_file):
    profiler = VideoProfilerFFprobe()
    res = profiler.probe(invalid_video_file)

    assert res.is_valid is False
    assert res.error_message is not None


def test_profiler_zero_byte_file(test_temp_dir):
    empty_file = os.path.join(test_temp_dir, "empty_video.mp4")
    with open(empty_file, "wb") as f:
        pass

    profiler = VideoProfilerFFprobe()
    res = profiler.probe(empty_file)

    assert res.is_valid is False
    assert "Zero-byte" in res.error_message or "empty" in res.error_message


# =====================================================================
# 2. Break-Even and ROI Decision Engine Tests
# =====================================================================

def test_roi_exact_formula_calculation():
    """Verify net savings formula matches specification exactly."""
    params = CostModelParameters(
        storage_cost_per_gb_month=0.020,
        read_io_cost_per_gb=0.005,
        transcode_compute_cost_per_hour=0.060,
        io_transfer_speed_mb_per_s=100.0,
        expected_vlm_passes=10,
        retention_months=6.0,
        bypass_bitrate_mb_per_min=80.0
    )
    engine = ConversionDecisionEngine(default_params=params)

    # Synthetic mock high-bitrate video: 1000 MB, 60s, raw codec
    meta = VideoProbeResult(
        video_id="test_raw_huge",
        path="huge.mov",
        file_size_bytes=1000 * 1024 * 1024,
        file_size_mb=1000.0,
        duration_s=60.0,
        bitrate_kbps=133333.0,  # ~133 Mbps (huge)
        width=1920,
        height=1080,
        fps=30.0,
        codec="prores",  # Legacy non-efficient codec
        is_valid=True
    )

    roi = engine.evaluate_estimate(meta, profile_name="h264_baseline", params=params)

    # Expected analytical math:
    # r = 0.55 -> delta_mb = 1000 * 0.45 = 450 MB -> delta_gb = 450 / 1024 GB
    delta_gb = 450.0 / 1024.0
    expected_storage_savings = delta_gb * 0.020 * 6.0
    expected_io_savings = delta_gb * 0.005 * 10
    # speed factor = 3.0x -> time_to_encode = 60 / 3.0 = 20s
    expected_encoding_cost = (20.0 / 3600.0) * 0.060
    expected_net_savings = expected_storage_savings + expected_io_savings - expected_encoding_cost

    assert roi.is_measured is False
    assert abs(roi.storage_savings_usd - expected_storage_savings) < 1e-4
    assert abs(roi.io_savings_usd - expected_io_savings) < 1e-4
    assert abs(roi.encoding_cost_usd - expected_encoding_cost) < 1e-4
    assert abs(roi.net_financial_savings_usd - expected_net_savings) < 1e-4
    assert roi.recommendation == DecisionRecommendation.TRANSCODE
    assert roi.net_financial_savings_usd > 0.0


def test_smart_bypass_already_efficient():
    """Video with bitrate < 80 MB/min already in H.264 should bypass transcoding."""
    params = CostModelParameters(bypass_bitrate_mb_per_min=80.0)
    engine = ConversionDecisionEngine(default_params=params)

    # 10 MB for 60s in H.264 -> 10 MB/min (well below 80 MB/min)
    meta = VideoProbeResult(
        video_id="stream_efficient",
        path="stream.mp4",
        file_size_bytes=10 * 1024 * 1024,
        file_size_mb=10.0,
        duration_s=60.0,
        bitrate_kbps=1365.0,  # ~1.3 Mbps
        width=1280,
        height=720,
        fps=30.0,
        codec="h264",
        is_valid=True
    )

    roi = engine.evaluate_estimate(meta, profile_name="h264_baseline", params=params)
    assert roi.recommendation == DecisionRecommendation.BYPASS_ALREADY_EFFICIENT
    assert "below 80.0 MB/min threshold" in roi.rationale


def test_keep_original_uneconomic():
    """Tiny video where transcode compute exceeds microscopic savings should be kept as original."""
    params = CostModelParameters(
        storage_cost_per_gb_month=0.001,
        transcode_compute_cost_per_hour=1.00,  # Expensive compute
        expected_vlm_passes=1,
        retention_months=0.1
    )
    engine = ConversionDecisionEngine(default_params=params)

    meta = VideoProbeResult(
        video_id="tiny_clip",
        path="tiny.avi",
        file_size_bytes=500 * 1024,  # 0.5 MB
        file_size_mb=0.5,
        duration_s=300.0,            # Long duration -> high encode time
        bitrate_kbps=13.0,
        width=320,
        height=240,
        fps=10.0,
        codec="mjpeg",
        is_valid=True
    )

    roi = engine.evaluate_estimate(meta, profile_name="h264_baseline", params=params)
    assert roi.recommendation == DecisionRecommendation.KEEP_ORIGINAL_UNECONOMIC
    assert roi.net_financial_savings_usd < 0.0


def test_measured_roi_evaluation():
    """Verify post-transcode empirical measurement correctly labels is_measured."""
    engine = ConversionDecisionEngine()

    orig = VideoProbeResult(
        video_id="v1", path="orig.mp4", file_size_mb=100.0, duration_s=10.0, is_valid=True
    )
    trans = VideoProbeResult(
        video_id="v1_h264", path="trans.mp4", file_size_mb=40.0, duration_s=10.0, is_valid=True
    )

    roi = engine.evaluate_measured(orig, trans, actual_encode_time_s=2.5)

    assert roi.is_measured is True
    assert roi.original_size_mb == 100.0
    assert roi.output_size_mb == 40.0
    assert roi.size_reduction_mb == 60.0
    assert roi.compression_ratio == 0.40
    assert roi.time_to_encode_s == 2.5
    assert roi.recommendation == DecisionRecommendation.TRANSCODE


# =====================================================================
# 3. Content-Aware Profile Adaptation Tests
# =====================================================================

def test_content_aware_sports_motion():
    base = get_profile("h264_baseline")
    assert base.crf == 23

    meta = VideoProbeResult(
        video_id="sports_vid", path="sports.mp4", width=1280, height=720, fps=30.0, is_valid=True
    )
    adapted = adapt_profile_for_content(base, meta, content_genre="sports")

    # High motion sports gets lower CRF for quality
    assert adapted.crf == 21


def test_content_aware_meetings_ocr_preservation():
    base = get_profile("h264_baseline")
    meta = VideoProbeResult(
        video_id="meetings_vid", path="meeting.mp4", width=1280, height=720, fps=25.0, is_valid=True
    )
    adapted = adapt_profile_for_content(base, meta, content_genre="meetings")

    assert adapted.tune == "stillimage"
    assert adapted.crf <= 22


def test_content_aware_4k_downscaling():
    base = get_profile("h264_baseline")
    meta = VideoProbeResult(
        video_id="4k_vid", path="4k.mp4", width=3840, height=2160, fps=60.0, is_valid=True
    )
    adapted = adapt_profile_for_content(base, meta, content_genre="general")

    assert adapted.max_width == 1920
    assert adapted.max_height == 1080
    assert adapted.max_fps == 30.0


def test_content_aware_low_res_preservation():
    base = get_profile("h264_baseline")
    meta = VideoProbeResult(
        video_id="360p_vid", path="low.mp4", width=480, height=360, fps=24.0, is_valid=True
    )
    adapted = adapt_profile_for_content(base, meta)

    assert adapted.max_width is None
    assert adapted.max_height is None


# =====================================================================
# 4. Objective Quality Evaluation Tests
# =====================================================================

def test_psnr_and_ssim_identical_images():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = (255, 128, 64)

    psnr = calculate_psnr(img, img)
    ssim = calculate_ssim(img, img)
    edge = calculate_edge_preservation(img, img)

    assert psnr >= 99.0
    assert abs(ssim - 1.0) < 1e-4
    assert abs(edge - 1.0) < 1e-3


def test_psnr_and_ssim_noisy_images():
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 120
    # Add small perturbation
    noise = np.random.randint(-10, 10, (100, 100, 3), dtype=np.int16)
    img2 = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)

    assert 20.0 < psnr < 50.0
    assert 0.5 < ssim < 1.0


# =====================================================================
# 5. Non-Destructive Transcoding Execution Tests
# =====================================================================

def test_transcode_non_destructive_guarantee(sample_sports_video, test_temp_dir):
    """Ensure original media file is NEVER modified or overwritten."""
    orig_mtime = os.path.getmtime(sample_sports_video)
    orig_size = os.path.getsize(sample_sports_video)

    engine = TranscodeEngine(default_output_dir=test_temp_dir)
    profile = get_profile("opencv_fallback")

    out_path = os.path.join(test_temp_dir, "transcoded_sample.mp4")
    res = engine.transcode(sample_sports_video, profile, output_path=out_path, duration_s=1.0)

    assert res.success is True
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0

    # Verify original master is byte-identical and unmodified
    assert os.path.getmtime(sample_sports_video) == orig_mtime
    assert os.path.getsize(sample_sports_video) == orig_size


def test_measure_video_quality_on_transcoded_file(sample_sports_video, test_temp_dir):
    engine = TranscodeEngine(default_output_dir=test_temp_dir)
    profile = get_profile("opencv_fallback")
    out_path = os.path.join(test_temp_dir, "quality_check_out.mp4")

    res = engine.transcode(sample_sports_video, profile, output_path=out_path)
    assert res.success is True

    q_report = measure_video_quality(sample_sports_video, out_path, num_eval_frames=3)
    assert q_report.frames_evaluated > 0
    assert q_report.psnr_db > 25.0
    assert 0.70 <= q_report.ssim <= 1.0


# =====================================================================
# 6. Multi-Codec Benchmarking Tests
# =====================================================================

def test_codec_benchmark_runner(sample_sports_video, test_temp_dir):
    runner = CodecBenchmarkRunner(output_dir=test_temp_dir)
    scorecard = runner.benchmark(sample_sports_video, profile_names=["opencv_fallback"])

    assert scorecard.duration_s > 0.0
    assert len(scorecard.entries) == 1
    entry = scorecard.entries[0]
    assert entry.success is True
    assert entry.compression_ratio > 0.0
    assert entry.psnr_db > 0.0
    assert entry.ssim > 0.0
    assert scorecard.recommended_profile == "opencv_fallback"

    table_md = scorecard.format_markdown_table()
    assert "| Codec Profile |" in table_md
    assert "opencv_fallback" in table_md


# =====================================================================
# 7. CLI Integration Tests
# =====================================================================

def test_cli_analyze_conversion_dry_run(sample_sports_video):
    result = subprocess.run(
        [sys.executable, "src/main.py", "analyze-conversion", "--video", sample_sports_video, "--dry-run"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "VIDEO CONVERSION & BREAK-EVEN ANALYSIS" in result.stdout
    assert "Recommendation:" in result.stdout
    assert "PRE-TRANSCODE ANALYTICAL ESTIMATE" in result.stdout


def test_cli_analyze_conversion_json(sample_sports_video):
    result = subprocess.run(
        [sys.executable, "src/main.py", "analyze-conversion", "--video", sample_sports_video, "--dry-run", "--json"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "video" in data
    assert "roi_estimate" in data
    assert "recommendation" in data["roi_estimate"]


def test_cli_analyze_conversion_transcode(sample_sports_video, test_temp_dir):
    result = subprocess.run(
        [
            sys.executable, "src/main.py", "analyze-conversion",
            "--video", sample_sports_video,
            "--transcode", "--force",
            "--profile", "opencv_fallback",
            "--output-dir", test_temp_dir
        ],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "POST-TRANSCODE EMPIRICAL MEASUREMENT" in result.stdout
    assert "Transcode Success:     True" in result.stdout
    assert "Derived Media File:" in result.stdout
