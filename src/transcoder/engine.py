"""
Safe Transcoding Execution Engine (Task 3)
Builds and executes parameterized FFmpeg commands using argument lists (shell=False)
with strict timeouts and non-destructive output guarantees.
Provides an OpenCV VideoWriter fallback for environments lacking FFmpeg binaries.
"""

from dataclasses import asdict, dataclass
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import cv2

from src.transcoder.profiles import CodecProfile, get_profile
from src.transcoder.profiler import VideoProbeResult

logger = logging.getLogger("aintropy-transcoder.engine")


@dataclass
class TranscodeResult:
    """Outcome of a transcoding operation."""
    success: bool
    input_path: str
    output_path: str
    backend: str
    profile_name: str
    elapsed_time_s: float
    original_size_bytes: int
    output_size_bytes: int
    compression_ratio: float
    speed_factor: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TranscodeEngine:
    """Executes safe, non-destructive media transcoding with FFmpeg or OpenCV fallback."""

    def __init__(
        self,
        default_output_dir: str = "derived",
        timeout_s: float = 120.0,
        prefer_ffmpeg: bool = True
    ):
        self.default_output_dir = default_output_dir
        self.timeout_s = timeout_s
        self.prefer_ffmpeg = prefer_ffmpeg
        self._ffmpeg_available = shutil.which("ffmpeg") is not None

    @property
    def is_ffmpeg_available(self) -> bool:
        return self._ffmpeg_available

    def generate_output_path(
        self,
        input_path: str,
        profile: CodecProfile,
        output_dir: Optional[str] = None
    ) -> str:
        """Construct deterministic output filepath in dedicated derived directory."""
        target_dir = output_dir or self.default_output_dir
        os.makedirs(target_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = ".mp4"
        if "hevc" in profile.video_codec or "265" in profile.video_codec:
            ext = ".mp4"
        elif "av1" in profile.video_codec:
            ext = ".mp4"

        out_name = f"{base_name}_{profile.name}{ext}"
        return os.path.join(target_dir, out_name)

    def build_ffmpeg_cmd(
        self,
        input_path: str,
        output_path: str,
        profile: CodecProfile
    ) -> List[str]:
        """
        Build argument list for FFmpeg execution.
        Security Note: Always an argument list; never executed via shell=True.
        """
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output only inside derived/, never original
            "-i", input_path,
            "-c:v", profile.video_codec,
            "-crf", str(profile.crf),
            "-preset", profile.preset,
            "-pix_fmt", profile.pixel_format,
        ]

        # Content-aware filters (scaling, fps clamp)
        filters = []
        if profile.max_width and profile.max_height:
            # Scale maintaining aspect ratio, ensuring even dimensions for codecs
            filters.append(
                f"scale='min({profile.max_width},iw)':-2"
            )
        if profile.max_fps:
            filters.append(f"fps={profile.max_fps}")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        if profile.tune:
            cmd.extend(["-tune", profile.tune])

        if profile.audio_codec and profile.audio_codec != "none":
            cmd.extend(["-c:a", profile.audio_codec, "-b:a", profile.audio_bitrate])
        else:
            cmd.append("-an")

        cmd.extend(profile.extra_ffmpeg_args)
        cmd.append(output_path)
        return cmd

    def transcode(
        self,
        input_path: str,
        profile: CodecProfile,
        output_path: Optional[str] = None,
        duration_s: float = 0.0
    ) -> TranscodeResult:
        """
        Execute transcoding using FFmpeg or OpenCV fallback.
        Guarantee: Never overwrites or modifies input_path.
        """
        if not os.path.exists(input_path):
            return TranscodeResult(
                success=False,
                input_path=input_path,
                output_path="",
                backend="none",
                profile_name=profile.name,
                elapsed_time_s=0.0,
                original_size_bytes=0,
                output_size_bytes=0,
                compression_ratio=1.0,
                speed_factor=0.0,
                error_message=f"Input file not found: {input_path}"
            )

        orig_size = os.path.getsize(input_path)
        target_output = output_path or self.generate_output_path(input_path, profile)

        # Safety check: Prevent writing directly over original video
        if os.path.abspath(input_path) == os.path.abspath(target_output):
            target_output = self.generate_output_path(input_path, profile, output_dir="derived")

        os.makedirs(os.path.dirname(os.path.abspath(target_output)), exist_ok=True)

        # 1. Try FFmpeg execution if available
        if self.prefer_ffmpeg and self._ffmpeg_available:
            try:
                return self._transcode_via_ffmpeg(input_path, target_output, profile, orig_size, duration_s)
            except Exception as e:
                logger.warning("FFmpeg transcode failed (%s); falling back to OpenCV VideoWriter", str(e))

        # 2. Fallback to OpenCV VideoWriter
        return self._transcode_via_opencv(input_path, target_output, profile, orig_size, duration_s)

    def _transcode_via_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        profile: CodecProfile,
        orig_size: int,
        duration_s: float
    ) -> TranscodeResult:
        """Run FFmpeg subprocess with argument list and strict timeout."""
        cmd = self.build_ffmpeg_cmd(input_path, output_path, profile)
        start_t = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
                timeout=self.timeout_s,
                check=False
            )
            elapsed_s = max(0.001, time.perf_counter() - start_t)

            if proc.returncode != 0:
                err_text = proc.stderr.strip() or f"FFmpeg failed with code {proc.returncode}"
                return TranscodeResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    backend="ffmpeg",
                    profile_name=profile.name,
                    elapsed_time_s=round(elapsed_s, 2),
                    original_size_bytes=orig_size,
                    output_size_bytes=0,
                    compression_ratio=1.0,
                    speed_factor=0.0,
                    error_message=f"FFmpeg error: {err_text[:300]}"
                )

            out_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            ratio = round(out_size / orig_size, 4) if orig_size > 0 else 1.0
            speed = round(duration_s / elapsed_s, 2) if duration_s > 0 else round(1.0 / elapsed_s, 2)

            return TranscodeResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                backend="ffmpeg",
                profile_name=profile.name,
                elapsed_time_s=round(elapsed_s, 2),
                original_size_bytes=orig_size,
                output_size_bytes=out_size,
                compression_ratio=ratio,
                speed_factor=speed
            )

        except subprocess.TimeoutExpired:
            elapsed_s = time.perf_counter() - start_t
            if os.path.exists(output_path):
                os.remove(output_path)
            return TranscodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend="ffmpeg",
                profile_name=profile.name,
                elapsed_time_s=round(elapsed_s, 2),
                original_size_bytes=orig_size,
                output_size_bytes=0,
                compression_ratio=1.0,
                speed_factor=0.0,
                error_message=f"Transcode timed out after {self.timeout_s}s"
            )

    def _transcode_via_opencv(
        self,
        input_path: str,
        output_path: str,
        profile: CodecProfile,
        orig_size: int,
        duration_s: float
    ) -> TranscodeResult:
        """
        OpenCV VideoWriter fallback for offline or headless environments without FFmpeg.
        Performs genuine frame decoding, optional resolution scaling, and re-encoding.
        """
        start_t = time.perf_counter()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return TranscodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend="opencv",
                profile_name=profile.name,
                elapsed_time_s=0.0,
                original_size_bytes=orig_size,
                output_size_bytes=0,
                compression_ratio=1.0,
                speed_factor=0.0,
                error_message="Could not open input video with OpenCV backend"
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

        out_w, out_h = orig_w, orig_h
        if profile.max_width and profile.max_height and (orig_w > profile.max_width or orig_h > profile.max_height):
            scale = min(profile.max_width / orig_w, profile.max_height / orig_h)
            out_w = int(orig_w * scale)
            out_h = int(orig_h * scale)
            # Ensure dimensions are even
            out_w = out_w if out_w % 2 == 0 else out_w - 1
            out_h = out_h if out_h % 2 == 0 else out_h - 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            return TranscodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                backend="opencv",
                profile_name=profile.name,
                elapsed_time_s=0.0,
                original_size_bytes=orig_size,
                output_size_bytes=0,
                compression_ratio=1.0,
                speed_factor=0.0,
                error_message="Could not initialize OpenCV VideoWriter for output container"
            )

        frames_written = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if (out_w, out_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frames_written += 1

        cap.release()
        writer.release()
        elapsed_s = max(0.001, time.perf_counter() - start_t)

        out_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        ratio = round(out_size / orig_size, 4) if orig_size > 0 else 1.0
        dur = duration_s if duration_s > 0 else (frames_written / fps if fps > 0 else 1.0)
        speed = round(dur / elapsed_s, 2)

        return TranscodeResult(
            success=True,
            input_path=input_path,
            output_path=output_path,
            backend="opencv",
            profile_name=profile.name,
            elapsed_time_s=round(elapsed_s, 2),
            original_size_bytes=orig_size,
            output_size_bytes=out_size,
            compression_ratio=ratio,
            speed_factor=speed
        )
