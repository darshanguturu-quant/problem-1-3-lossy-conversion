"""
Video Profiler & Stream Metadata Extractor (Task 3)
Extracts container and stream metadata via FFprobe with graceful OpenCV fallback.
Handles missing, corrupted, or unsupported media files safely.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import cv2

logger = logging.getLogger("aintropy-transcoder.profiler")


@dataclass
class VideoProbeResult:
    """Detailed media stream and container profile."""
    video_id: str
    path: str
    file_size_bytes: int = 0
    file_size_mb: float = 0.0
    duration_s: float = 0.0
    bitrate_kbps: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = "unknown"
    format_name: str = "unknown"
    has_audio: bool = False
    audio_codec: str = "none"
    audio_channels: int = 0
    audio_sample_rate_hz: int = 0
    is_valid: bool = False
    probe_backend: str = "none"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoProfilerFFprobe:
    """Robust video inspection engine using FFprobe with OpenCV fallback."""

    def __init__(self, timeout_s: float = 30.0, prefer_ffprobe: bool = True):
        self.timeout_s = timeout_s
        self.prefer_ffprobe = prefer_ffprobe
        self._ffprobe_available = shutil.which("ffprobe") is not None

    @property
    def is_ffprobe_available(self) -> bool:
        """Check if ffprobe binary is present on system PATH."""
        return self._ffprobe_available

    def probe(self, video_path: str) -> VideoProbeResult:
        """Inspect video file and return structured probe result."""
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # 1. File existence validation
        if not os.path.exists(video_path):
            return VideoProbeResult(
                video_id=video_id,
                path=video_path,
                is_valid=False,
                error_message=f"File not found: {video_path}"
            )

        # 2. File size validation
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return VideoProbeResult(
                video_id=video_id,
                path=video_path,
                file_size_bytes=0,
                file_size_mb=0.0,
                is_valid=False,
                error_message="Zero-byte file or empty media container"
            )

        file_size_mb = round(file_size / (1024.0 * 1024.0), 3)

        # 3. Attempt FFprobe probe if available
        if self.prefer_ffprobe and self._ffprobe_available:
            try:
                res = self._probe_via_ffprobe(video_path, video_id, file_size, file_size_mb)
                if res.is_valid:
                    return res
                logger.warning("FFprobe probe returned invalid result for %s; falling back to OpenCV", video_path)
            except Exception as e:
                logger.warning("FFprobe probe failed for %s (%s); falling back to OpenCV", video_path, str(e))

        # 4. Fallback to OpenCV probe
        return self._probe_via_opencv(video_path, video_id, file_size, file_size_mb)

    def _probe_via_ffprobe(
        self, video_path: str, video_id: str, file_size: int, file_size_mb: float
    ) -> VideoProbeResult:
        """Execute ffprobe with argument list (shell=False) and parse JSON."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate,format_name:stream=codec_name,codec_type,width,height,r_frame_rate,channels,sample_rate",
            "-of", "json",
            video_path
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=self.timeout_s,
            check=False
        )

        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or f"ffprobe exited with code {proc.returncode}"
            return VideoProbeResult(
                video_id=video_id,
                path=video_path,
                file_size_bytes=file_size,
                file_size_mb=file_size_mb,
                is_valid=False,
                probe_backend="ffprobe",
                error_message=f"ffprobe error: {err_msg}"
            )

        data = json.loads(proc.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])

        video_stream = None
        audio_stream = None
        for s in streams:
            stype = s.get("codec_type")
            if stype == "video" and video_stream is None:
                video_stream = s
            elif stype == "audio" and audio_stream is None:
                audio_stream = s

        if not video_stream:
            return VideoProbeResult(
                video_id=video_id,
                path=video_path,
                file_size_bytes=file_size,
                file_size_mb=file_size_mb,
                is_valid=False,
                probe_backend="ffprobe",
                error_message="No video stream detected in container"
            )

        # Parse FPS
        fps_str = video_stream.get("r_frame_rate", "0/1")
        fps = 0.0
        if "/" in fps_str:
            num, den = fps_str.split("/", 1)
            try:
                num_f, den_f = float(num), float(den)
                if den_f > 0:
                    fps = round(num_f / den_f, 2)
            except ValueError:
                fps = 0.0
        else:
            try:
                fps = round(float(fps_str), 2)
            except ValueError:
                fps = 0.0

        # Duration
        duration_s = 0.0
        if "duration" in fmt:
            try:
                duration_s = round(float(fmt["duration"]), 2)
            except (ValueError, TypeError):
                duration_s = 0.0
        elif "duration" in video_stream:
            try:
                duration_s = round(float(video_stream["duration"]), 2)
            except (ValueError, TypeError):
                duration_s = 0.0

        # Bitrate
        bitrate_kbps = 0.0
        if "bit_rate" in fmt:
            try:
                bitrate_kbps = round(float(fmt["bit_rate"]) / 1000.0, 2)
            except (ValueError, TypeError):
                bitrate_kbps = 0.0
        if bitrate_kbps == 0.0 and duration_s > 0:
            bitrate_kbps = round((file_size * 8.0) / (duration_s * 1000.0), 2)

        # Audio stream info
        has_audio = audio_stream is not None
        audio_codec = audio_stream.get("codec_name", "none") if has_audio else "none"
        audio_channels = int(audio_stream.get("channels", 0)) if has_audio else 0
        audio_sr = int(audio_stream.get("sample_rate", 0)) if has_audio else 0

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "unknown")
        format_name = fmt.get("format_name", os.path.splitext(video_path)[1].lstrip(".")).lower()

        is_valid = (width > 0 and height > 0 and (duration_s > 0 or fps > 0))

        return VideoProbeResult(
            video_id=video_id,
            path=video_path,
            file_size_bytes=file_size,
            file_size_mb=file_size_mb,
            duration_s=duration_s,
            bitrate_kbps=bitrate_kbps,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            format_name=format_name,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_channels=audio_channels,
            audio_sample_rate_hz=audio_sr,
            is_valid=is_valid,
            probe_backend="ffprobe",
            error_message=None if is_valid else "Invalid or empty video dimensions"
        )

    def _probe_via_opencv(
        self, video_path: str, video_id: str, file_size: int, file_size_mb: float
    ) -> VideoProbeResult:
        """Inspect media using OpenCV VideoCapture backend."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoProbeResult(
                video_id=video_id,
                path=video_path,
                file_size_bytes=file_size,
                file_size_mb=file_size_mb,
                is_valid=False,
                probe_backend="opencv",
                error_message="Could not open video with media decoder (unsupported format or corrupted header)"
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        cap.release()

        # Deduce fourcc
        codec = "unknown"
        if fourcc_int != 0:
            try:
                fourcc_str = "".join([chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]).strip()
                if fourcc_str:
                    codec = fourcc_str.lower()
            except Exception:
                codec = "unknown"

        # Common codec normalization
        if codec in ["avc1", "h264", "x264"]:
            codec = "h264"
        elif codec in ["hev1", "hvc1", "hevc", "h265", "x265"]:
            codec = "hevc"
        elif codec in ["mp4v", "fmp4"]:
            codec = "mpeg4"
        elif codec == "unknown":
            ext = os.path.splitext(video_path)[1].lstrip(".").lower()
            if ext in ["mp4", "m4v"]:
                codec = "h264"  # Default assumption for standard MP4 containers
            elif ext in ["mkv", "webm"]:
                codec = "vp9"

        duration_s = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        bitrate_kbps = (file_size * 8.0 / (duration_s * 1000.0)) if duration_s > 0 else 0.0
        ext = os.path.splitext(video_path)[1].lstrip(".").lower()

        is_valid = (width > 0 and height > 0 and (duration_s > 0 or frame_count > 0))

        return VideoProbeResult(
            video_id=video_id,
            path=video_path,
            file_size_bytes=file_size,
            file_size_mb=file_size_mb,
            duration_s=round(duration_s, 2),
            bitrate_kbps=round(bitrate_kbps, 2),
            width=width,
            height=height,
            fps=round(fps, 2),
            codec=codec,
            format_name=ext or "mp4",
            has_audio=True,  # Heuristic assumption in headless OpenCV
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate_hz=44100,
            is_valid=is_valid,
            probe_backend="opencv",
            error_message=None if is_valid else "Could not decode frames or calculate video duration"
        )
