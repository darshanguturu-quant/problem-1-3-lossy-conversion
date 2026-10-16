"""
Task 3 — Lossy Conversion & Metadata-Driven ROI Engine
Core package exports for video profiling, ROI break-even analysis,
content-aware codec adaptation, transcoding, quality checking, and benchmarking.
"""

from src.transcoder.profiler import VideoProbeResult, VideoProfilerFFprobe
from src.transcoder.roi import (
    CostModelParameters,
    ROIEstimate,
    ConversionDecisionEngine,
    DecisionRecommendation,
)
from src.transcoder.profiles import (
    CodecProfile,
    get_profile,
    adapt_profile_for_content,
    AVAILABLE_PROFILES,
)
from src.transcoder.quality import QualityReport, measure_video_quality
from src.transcoder.engine import TranscodeEngine, TranscodeResult
from src.transcoder.benchmark import CodecBenchmarkRunner, CodecBenchmarkScorecard

__all__ = [
    "VideoProbeResult",
    "VideoProfilerFFprobe",
    "CostModelParameters",
    "ROIEstimate",
    "ConversionDecisionEngine",
    "DecisionRecommendation",
    "CodecProfile",
    "get_profile",
    "adapt_profile_for_content",
    "AVAILABLE_PROFILES",
    "QualityReport",
    "measure_video_quality",
    "TranscodeEngine",
    "TranscodeResult",
    "CodecBenchmarkRunner",
    "CodecBenchmarkScorecard",
]
