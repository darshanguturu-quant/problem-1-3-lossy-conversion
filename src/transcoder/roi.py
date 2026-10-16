"""
Break-Even and ROI Decision Engine (Task 3)
Implements analytical and empirical break-even equations:
Net savings = Original processing/I/O cost + storage savings - Compressed processing/I/O cost - Encoding cost

Defines explicit cost parameters, units, assumptions, and recommendation logic.
Clearly distinguishes analytical estimates from empirical measurements.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional

from src.transcoder.profiler import VideoProbeResult


class DecisionRecommendation(str, Enum):
    TRANSCODE = "TRANSCODE"
    BYPASS_ALREADY_EFFICIENT = "BYPASS_ALREADY_EFFICIENT"
    KEEP_ORIGINAL_UNECONOMIC = "KEEP_ORIGINAL_UNECONOMIC"
    INVALID_MEDIA = "INVALID_MEDIA"


@dataclass
class CostModelParameters:
    """
    Cost model parameters and unit assumptions for break-even calculation.
    
    Units:
      - storage_cost_per_gb_month: USD per Gigabyte per Month (AWS S3 Standard ~$0.020/GB/month)
      - read_io_cost_per_gb: USD per Gigabyte transferred / read across AZ/service boundaries (~$0.005/GB)
      - transcode_compute_cost_per_hour: USD per hour for 4-core CPU compute instance (~$0.060/hour)
      - io_transfer_speed_mb_per_s: Megabytes per second for disk/network I/O pipeline (~150 MB/s)
      - expected_vlm_passes: Number of downstream retrieval / VLM captioning queries over the video lifetime
      - retention_months: Storage retention period in months
      - bypass_bitrate_mb_per_min: Bitrate threshold (MB/min) below which modern codecs bypass transcoding
    
    Important Downstream Nuance:
      Downstream VLM inference model FLOPs (e.g. Qwen-VL, Gemini, CLIP) are NOT reduced by lossy video
      compression because the VLM internally samples and resizes keyframe patches (e.g. to 224x224 or 384x384)
      regardless of video bitrate. Therefore, financial and latency savings originate strictly from:
        1. Reduced long-term cloud storage (GB-months)
        2. Reduced network transfer and egress bandwidth (GB transferred per pass)
        3. Reduced disk read and memory buffer allocation time (I/O latency per pass)
    """
    storage_cost_per_gb_month: float = 0.020
    read_io_cost_per_gb: float = 0.005
    transcode_compute_cost_per_hour: float = 0.060
    io_transfer_speed_mb_per_s: float = 150.0
    expected_vlm_passes: int = 5
    retention_months: float = 3.0
    bypass_bitrate_mb_per_min: float = 80.0
    vlm_inference_cost_affected: bool = False


# Nominal codec priors for pre-transcode analytical estimation
NOMINAL_CODEC_PRIORS = {
    "h264_baseline": {
        "expected_compression_ratio": 0.55,  # 45% size reduction
        "expected_speed_factor": 3.0,        # 3.0x real-time on CPU (e.g. 10s video encodes in 3.3s)
    },
    "h265_storage": {
        "expected_compression_ratio": 0.38,  # 62% size reduction
        "expected_speed_factor": 1.6,        # 1.6x real-time on CPU
    },
    "av1_efficient": {
        "expected_compression_ratio": 0.30,  # 70% size reduction
        "expected_speed_factor": 0.75,       # 0.75x real-time on CPU (slower, higher compression)
    },
    "opencv_fallback": {
        "expected_compression_ratio": 0.65,
        "expected_speed_factor": 2.5,
    }
}


@dataclass
class ROIEstimate:
    """Structured breakdown of financial and time break-even evaluation."""
    is_measured: bool  # True if calculated on post-transcode measured data; False if analytical estimate
    original_size_mb: float
    output_size_mb: float
    size_reduction_mb: float
    compression_ratio: float
    
    # Financial breakdown (USD)
    storage_savings_usd: float
    original_io_cost_usd: float
    compressed_io_cost_usd: float
    io_savings_usd: float
    encoding_cost_usd: float
    net_financial_savings_usd: float
    
    # Time breakdown (seconds)
    time_to_encode_s: float
    original_io_time_s: float
    compressed_io_time_s: float
    io_time_savings_s: float
    net_time_savings_s: float
    
    # Decision metrics
    break_even_passes: float
    recommendation: DecisionRecommendation
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommendation"] = self.recommendation.value
        return d


class ConversionDecisionEngine:
    """
    Evaluates whether an incoming video should be converted to an optimized secondary
    representation based on the net savings break-even formulation.
    """

    def __init__(self, default_params: Optional[CostModelParameters] = None):
        self.params = default_params or CostModelParameters()

    def evaluate_estimate(
        self,
        meta: VideoProbeResult,
        profile_name: str = "h264_baseline",
        params: Optional[CostModelParameters] = None
    ) -> ROIEstimate:
        """
        Analytical pre-transcode evaluation using nominal profile assumptions.
        Distinguishes estimated values from empirical measurements.
        """
        p = params or self.params

        if not meta.is_valid:
            return ROIEstimate(
                is_measured=False,
                original_size_mb=meta.file_size_mb,
                output_size_mb=meta.file_size_mb,
                size_reduction_mb=0.0,
                compression_ratio=1.0,
                storage_savings_usd=0.0,
                original_io_cost_usd=0.0,
                compressed_io_cost_usd=0.0,
                io_savings_usd=0.0,
                encoding_cost_usd=0.0,
                net_financial_savings_usd=0.0,
                time_to_encode_s=0.0,
                original_io_time_s=0.0,
                compressed_io_time_s=0.0,
                io_time_savings_s=0.0,
                net_time_savings_s=0.0,
                break_even_passes=float("inf"),
                recommendation=DecisionRecommendation.INVALID_MEDIA,
                rationale=f"Invalid media file: {meta.error_message or 'Unknown error'}"
            )

        # 1. Check Smart Bypass criterion:
        # If video bitrate < 80 MB/min and codec is already modern (h264/h265/av1/vp9)
        bitrate_mb_per_min = (meta.bitrate_kbps * 60.0) / (8.0 * 1024.0) if meta.bitrate_kbps > 0 else (
            (meta.file_size_mb * 60.0 / meta.duration_s) if meta.duration_s > 0 else 0.0
        )
        modern_codecs = {"h264", "hevc", "h265", "av1", "vp9"}
        is_already_efficient = (
            bitrate_mb_per_min > 0 and
            bitrate_mb_per_min < p.bypass_bitrate_mb_per_min and
            meta.codec.lower() in modern_codecs
        )

        priors = NOMINAL_CODEC_PRIORS.get(profile_name, NOMINAL_CODEC_PRIORS["h264_baseline"])
        exp_r = priors["expected_compression_ratio"]
        exp_v = priors["expected_speed_factor"]

        # Calculate sizes
        orig_mb = meta.file_size_mb
        out_mb = round(orig_mb * exp_r, 3)
        delta_mb = round(orig_mb - out_mb, 3)

        # Time to encode estimate
        time_to_encode_s = round(meta.duration_s / exp_v, 2) if exp_v > 0 else 0.0

        # I/O transfer time estimates
        io_speed = max(p.io_transfer_speed_mb_per_s, 1.0)
        orig_io_time_s = round(p.expected_vlm_passes * (orig_mb / io_speed), 3)
        comp_io_time_s = round(p.expected_vlm_passes * (out_mb / io_speed), 3)
        io_time_savings_s = round(orig_io_time_s - comp_io_time_s, 3)
        net_time_savings_s = round(io_time_savings_s - time_to_encode_s, 3)

        # Financial cost calculations
        delta_gb = delta_mb / 1024.0
        orig_gb = orig_mb / 1024.0
        out_gb = out_mb / 1024.0

        storage_savings = delta_gb * p.storage_cost_per_gb_month * p.retention_months
        orig_io_cost = orig_gb * p.read_io_cost_per_gb * p.expected_vlm_passes
        comp_io_cost = out_gb * p.read_io_cost_per_gb * p.expected_vlm_passes
        io_savings = orig_io_cost - comp_io_cost

        encoding_cost = (time_to_encode_s / 3600.0) * p.transcode_compute_cost_per_hour
        net_fin_savings = storage_savings + io_savings - encoding_cost

        # Calculate break-even pass count
        io_savings_per_pass = (delta_gb * p.read_io_cost_per_gb) + (delta_mb / io_speed) * (p.transcode_compute_cost_per_hour / 3600.0)
        if storage_savings >= encoding_cost:
            break_even_passes = 0.0
        elif io_savings_per_pass > 0:
            break_even_passes = round((encoding_cost - storage_savings) / io_savings_per_pass, 2)
        else:
            break_even_passes = float("inf")

        # Recommendation synthesis
        if is_already_efficient:
            recommendation = DecisionRecommendation.BYPASS_ALREADY_EFFICIENT
            rationale = (
                f"Video is already encoded in efficient codec '{meta.codec}' at {bitrate_mb_per_min:.1f} MB/min "
                f"(below {p.bypass_bitrate_mb_per_min} MB/min threshold). Transcoding overhead exceeds marginal gains."
            )
        elif net_fin_savings > 0 or net_time_savings_s > 0:
            recommendation = DecisionRecommendation.TRANSCODE
            rationale = (
                f"Positive ROI: Estimated net financial savings of ${net_fin_savings:.4f} over {p.retention_months} months "
                f"with {p.expected_vlm_passes} passes (break-even at {break_even_passes} passes)."
            )
        else:
            recommendation = DecisionRecommendation.KEEP_ORIGINAL_UNECONOMIC
            rationale = (
                f"Uneconomic: Transcode compute cost (${encoding_cost:.4f}) exceeds combined storage/IO savings "
                f"(${storage_savings + io_savings:.4f}). Retaining original master without transcoding."
            )

        return ROIEstimate(
            is_measured=False,
            original_size_mb=orig_mb,
            output_size_mb=out_mb,
            size_reduction_mb=delta_mb,
            compression_ratio=exp_r,
            storage_savings_usd=round(storage_savings, 5),
            original_io_cost_usd=round(orig_io_cost, 5),
            compressed_io_cost_usd=round(comp_io_cost, 5),
            io_savings_usd=round(io_savings, 5),
            encoding_cost_usd=round(encoding_cost, 5),
            net_financial_savings_usd=round(net_fin_savings, 5),
            time_to_encode_s=time_to_encode_s,
            original_io_time_s=orig_io_time_s,
            compressed_io_time_s=comp_io_time_s,
            io_time_savings_s=io_time_savings_s,
            net_time_savings_s=net_time_savings_s,
            break_even_passes=break_even_passes,
            recommendation=recommendation,
            rationale=rationale
        )

    def evaluate_measured(
        self,
        orig_meta: VideoProbeResult,
        transcoded_meta: VideoProbeResult,
        actual_encode_time_s: float,
        params: Optional[CostModelParameters] = None
    ) -> ROIEstimate:
        """
        Empirical post-transcode evaluation using actual measured file sizes and wall times.
        """
        p = params or self.params
        orig_mb = orig_meta.file_size_mb
        out_mb = transcoded_meta.file_size_mb
        delta_mb = round(orig_mb - out_mb, 3)
        actual_r = round(out_mb / orig_mb, 4) if orig_mb > 0 else 1.0

        io_speed = max(p.io_transfer_speed_mb_per_s, 1.0)
        orig_io_time_s = round(p.expected_vlm_passes * (orig_mb / io_speed), 3)
        comp_io_time_s = round(p.expected_vlm_passes * (out_mb / io_speed), 3)
        io_time_savings_s = round(orig_io_time_s - comp_io_time_s, 3)
        net_time_savings_s = round(io_time_savings_s - actual_encode_time_s, 3)

        delta_gb = delta_mb / 1024.0
        orig_gb = orig_mb / 1024.0
        out_gb = out_mb / 1024.0

        storage_savings = delta_gb * p.storage_cost_per_gb_month * p.retention_months
        orig_io_cost = orig_gb * p.read_io_cost_per_gb * p.expected_vlm_passes
        comp_io_cost = out_gb * p.read_io_cost_per_gb * p.expected_vlm_passes
        io_savings = orig_io_cost - comp_io_cost

        encoding_cost = (actual_encode_time_s / 3600.0) * p.transcode_compute_cost_per_hour
        net_fin_savings = storage_savings + io_savings - encoding_cost

        io_savings_per_pass = (delta_gb * p.read_io_cost_per_gb) + (delta_mb / io_speed) * (p.transcode_compute_cost_per_hour / 3600.0)
        if storage_savings >= encoding_cost:
            break_even_passes = 0.0
        elif io_savings_per_pass > 0:
            break_even_passes = round((encoding_cost - storage_savings) / io_savings_per_pass, 2)
        else:
            break_even_passes = float("inf")

        if net_fin_savings > 0 or net_time_savings_s > 0:
            recommendation = DecisionRecommendation.TRANSCODE
            rationale = (
                f"Measured ROI is positive: Actual size reduced by {delta_mb} MB ({((1 - actual_r) * 100):.1f}%), "
                f"yielding ${net_fin_savings:.4f} net financial savings."
            )
        else:
            recommendation = DecisionRecommendation.KEEP_ORIGINAL_UNECONOMIC
            rationale = (
                f"Measured ROI is negative: Encoding cost (${encoding_cost:.4f}) outweighed measured savings "
                f"(${storage_savings + io_savings:.4f})."
            )

        return ROIEstimate(
            is_measured=True,
            original_size_mb=orig_mb,
            output_size_mb=out_mb,
            size_reduction_mb=delta_mb,
            compression_ratio=actual_r,
            storage_savings_usd=round(storage_savings, 5),
            original_io_cost_usd=round(orig_io_cost, 5),
            compressed_io_cost_usd=round(comp_io_cost, 5),
            io_savings_usd=round(io_savings, 5),
            encoding_cost_usd=round(encoding_cost, 5),
            net_financial_savings_usd=round(net_fin_savings, 5),
            time_to_encode_s=round(actual_encode_time_s, 2),
            original_io_time_s=orig_io_time_s,
            compressed_io_time_s=comp_io_time_s,
            io_time_savings_s=io_time_savings_s,
            net_time_savings_s=net_time_savings_s,
            break_even_passes=break_even_passes,
            recommendation=recommendation,
            rationale=rationale
        )
