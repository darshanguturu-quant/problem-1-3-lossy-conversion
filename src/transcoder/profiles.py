"""
Codec Profiles & Content-Aware Adaptation (Task 3)
Defines standardized encoding profiles (H.264, H.265, AV1, OpenCV fallback)
and adapts parameters based on resolution, framerate, motion dynamics, and OCR text needs.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.transcoder.profiler import VideoProbeResult


@dataclass
class CodecProfile:
    """Parameters defining a video transcoding configuration."""
    name: str
    video_codec: str
    crf: int
    preset: str
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"
    tune: Optional[str] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_fps: Optional[float] = None
    extra_ffmpeg_args: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Standard baseline profiles
AVAILABLE_PROFILES: Dict[str, CodecProfile] = {
    "h264_baseline": CodecProfile(
        name="h264_baseline",
        video_codec="libx264",
        crf=23,
        preset="fast",
        pixel_format="yuv420p",
        audio_codec="aac",
        audio_bitrate="128k"
    ),
    "h265_storage": CodecProfile(
        name="h265_storage",
        video_codec="libx265",
        crf=26,
        preset="medium",
        pixel_format="yuv420p",
        audio_codec="aac",
        audio_bitrate="96k"
    ),
    "av1_efficient": CodecProfile(
        name="av1_efficient",
        video_codec="libsvtav1",
        crf=30,
        preset="6",
        pixel_format="yuv420p",
        audio_codec="aac",
        audio_bitrate="96k"
    ),
    "opencv_fallback": CodecProfile(
        name="opencv_fallback",
        video_codec="mp4v",
        crf=23,
        preset="default",
        pixel_format="bgr24",
        audio_codec="none"
    )
}


def get_profile(profile_name: str) -> CodecProfile:
    """Retrieve profile by name with fallback to h264_baseline."""
    if profile_name in AVAILABLE_PROFILES:
        base = AVAILABLE_PROFILES[profile_name]
        # Return a copy so adaptations don't mutate global constants
        return CodecProfile(
            name=base.name,
            video_codec=base.video_codec,
            crf=base.crf,
            preset=base.preset,
            pixel_format=base.pixel_format,
            audio_codec=base.audio_codec,
            audio_bitrate=base.audio_bitrate,
            tune=base.tune,
            max_width=base.max_width,
            max_height=base.max_height,
            max_fps=base.max_fps,
            extra_ffmpeg_args=list(base.extra_ffmpeg_args)
        )
    return get_profile("h264_baseline")


def adapt_profile_for_content(
    base_profile: CodecProfile,
    meta: VideoProbeResult,
    content_genre: Optional[str] = None
) -> CodecProfile:
    """
    Content-Aware Adaptation:
    Adjusts CRF, resolution, and tuning parameters according to:
      1. Motion dynamics: Sports, gameplay, wildlife need lower CRF (higher quality) to avoid macro-blocking.
      2. Text/OCR preservation: Slides, meetings, news require sharp edges for OCR (tune stillimage / higher resolution).
      3. Resolution clamping: 4K/60fps media is downscaled to 1080p/30fps for downstream VLM ingestion.
      4. Low-resolution preservation: <= 480p videos preserve original dimensions without downscaling.
    """
    adapted = get_profile(base_profile.name)
    genre = (content_genre or "").lower()

    # 1. High-motion dynamic content (sports, gameplay, wildlife)
    if genre in ["sports", "gameplay", "wildlife"] or meta.fps >= 50.0:
        adapted.crf = max(18, adapted.crf - 2)  # Lower CRF = higher bitrate/quality for motion

    # 2. Structured dialogue / text / slides (meetings, news_lecture, podcasts)
    elif genre in ["meetings", "news_lecture", "podcast"] or "slide" in genre:
        adapted.crf = min(22, adapted.crf)
        adapted.tune = "stillimage"

    # 3. Resolution scaling for downstream VLM ingestion
    # If original video is > 1080p (e.g. 1440p, 4K), clamp to 1080p to prevent VLM memory blowups
    if meta.width > 1920 or meta.height > 1080:
        adapted.max_width = 1920
        adapted.max_height = 1080

    # Clamp framerate if high-FPS (> 30fps) for non-sports content
    if meta.fps > 30.0 and genre not in ["sports", "gameplay"]:
        adapted.max_fps = 30.0

    # 4. Low resolution preservation: <= 480p videos are never downscaled
    if meta.width <= 640 and meta.height <= 480:
        adapted.max_width = None
        adapted.max_height = None

    return adapted
