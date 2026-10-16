"""
Video Profiler & Frame Sampler Module
Extracts stream metadata, container parameters, and sparse visual frames.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class VideoMetadata:
    video_id: str
    path: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    bitrate_kbps: float = 0.0
    aspect_ratio: float = 0.0
    has_audio: bool = False
    audio_channels: int = 0
    file_size_bytes: int = 0
    format_name: str = "unknown"
    is_valid: bool = False
    error_message: Optional[str] = None


@dataclass
class CoarseSignals:
    hist_mean_r: float = 0.0
    hist_mean_g: float = 0.0
    hist_mean_b: float = 0.0
    hist_std_r: float = 0.0
    hist_std_g: float = 0.0
    hist_std_b: float = 0.0
    frame_diff_mean: float = 0.0
    frame_diff_std: float = 0.0
    cut_rate_estimate: float = 0.0


class VideoProfiler:
    """Robust, fast video inspection and frame extraction engine using OpenCV."""

    def __init__(self, default_coarse_frames: int = 4):
        self.default_coarse_frames = default_coarse_frames

    def probe(self, video_path: str) -> VideoMetadata:
        """Extract metadata parameters from container and streams."""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if not os.path.isfile(video_path):
            return VideoMetadata(
                video_id=video_id,
                path=video_path,
                is_valid=False,
                error_message=f"File not found: {video_path}"
            )

        file_size = os.path.getsize(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoMetadata(
                video_id=video_id,
                path=video_path,
                file_size_bytes=file_size,
                is_valid=False,
                error_message="Could not open video with media backend"
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        # Compute duration and bitrate
        duration_s = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
        bitrate_kbps = (file_size * 8.0 / (duration_s * 1000.0)) if duration_s > 0 else 0.0
        aspect_ratio = (width / height) if height > 0 else 0.0
        ext = os.path.splitext(video_path)[1].lstrip(".").lower()

        return VideoMetadata(
            video_id=video_id,
            path=video_path,
            duration_s=round(duration_s, 2),
            width=width,
            height=height,
            fps=round(fps, 2),
            frame_count=frame_count,
            bitrate_kbps=round(bitrate_kbps, 2),
            aspect_ratio=round(aspect_ratio, 3),
            has_audio=True,  # Default heuristic when ffprobe is absent
            audio_channels=2,
            file_size_bytes=file_size,
            format_name=ext or "mp4",
            is_valid=(duration_s > 0 or frame_count > 0)
        )

    def extract_frames(self, video_path: str, num_frames: int = 4) -> List[np.ndarray]:
        """Extract uniformly spaced frames using keyframe seeks."""
        if not os.path.isfile(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            # Fallback: sequential read of first few frames
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
            cap.release()
            return frames

        # Select target frame indices
        if frame_count <= num_frames:
            target_indices = list(range(frame_count))
        else:
            step = frame_count / float(num_frames + 1)
            target_indices = [int(step * (i + 1)) for i in range(num_frames)]

        frames = []
        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        return frames

    def compute_coarse_signals(self, frames: List[np.ndarray]) -> CoarseSignals:
        """Compute inexpensive color histogram statistics and frame differences."""
        if not frames:
            return CoarseSignals()

        # Compute RGB histograms per frame
        r_means, g_means, b_means = [], [], []
        r_stds, g_stds, b_stds = [], [], []

        for f in frames:
            # OpenCV BGR -> B, G, R
            b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
            b_means.append(float(np.mean(b)))
            g_means.append(float(np.mean(g)))
            r_means.append(float(np.mean(r)))
            b_stds.append(float(np.std(b)))
            g_stds.append(float(np.std(g)))
            r_stds.append(float(np.std(r)))

        # Frame difference metrics
        diffs = []
        cut_candidates = 0
        cut_threshold = 35.0  # Mean absolute grayscale difference threshold for a cut

        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            # Resize for fast comparison if large
            if gray1.shape[0] > 240:
                gray1 = cv2.resize(gray1, (320, 180))
                gray2 = cv2.resize(gray2, (320, 180))
            mad = float(np.mean(cv2.absdiff(gray1, gray2)))
            diffs.append(mad)
            if mad > cut_threshold:
                cut_candidates += 1

        diff_mean = float(np.mean(diffs)) if diffs else 0.0
        diff_std = float(np.std(diffs)) if diffs else 0.0
        cut_rate = (cut_candidates / len(diffs)) if diffs else 0.0

        return CoarseSignals(
            hist_mean_r=float(np.mean(r_means)),
            hist_mean_g=float(np.mean(g_means)),
            hist_mean_b=float(np.mean(b_means)),
            hist_std_r=float(np.mean(r_stds)),
            hist_std_g=float(np.mean(g_stds)),
            hist_std_b=float(np.mean(b_stds)),
            frame_diff_mean=diff_mean,
            frame_diff_std=diff_std,
            cut_rate_estimate=cut_rate
        )
