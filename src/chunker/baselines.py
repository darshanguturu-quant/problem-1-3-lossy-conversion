"""
Baseline chunking implementations for comparative evaluation (Task 4).
1. FixedDurationChunker: Uniform slicing across fixed time windows.
2. BlindSceneCutChunker: Naive splitting at every scene cut without duration constraints.
"""

import time
import uuid
from typing import List, Dict, Any, Optional

from src.chunker.types import ChildChunk, ParentSegment, ChunkManifest, BoundaryCandidate
from src.chunker.signals import MultiSignalDetector


class FixedDurationChunker:
    """
    Uniform fixed-duration baseline chunker.
    Blindly cuts media every `target_duration_s` seconds, ignoring underlying visual/speech content.
    """

    def __init__(self, target_duration_s: float = 30.0, overlap_ratio: float = 0.0):
        self.target_duration_s = max(1.0, float(target_duration_s))
        self.overlap_ratio = max(0.0, min(0.25, float(overlap_ratio)))

    def chunk(
        self,
        video_path: str,
        video_duration_s: float,
        video_id: Optional[str] = None,
        parent_duration_s: float = 180.0,
    ) -> ChunkManifest:
        """Executes uniform fixed-duration segmentation."""
        t0 = time.perf_counter()
        vid = video_id or (video_path.split("/")[-1].split("\\")[-1].split(".")[0])
        chunks: List[ChildChunk] = []

        if video_duration_s <= 0:
            return ChunkManifest(
                video_id=vid,
                video_path=video_path,
                total_duration_s=0.0,
                strategy="fixed_duration_baseline",
                policy_name="fixed_duration",
                num_chunks=0,
                num_parents=0,
            )

        step_s = self.target_duration_s * (1.0 - self.overlap_ratio)
        current_start = 0.0
        idx = 0

        while current_start < video_duration_s:
            current_end = min(video_duration_s, current_start + self.target_duration_s)
            dur = current_end - current_start
            chunk_id = f"{vid}_c{idx:03d}"

            # Dummy parent assignment (every parent_duration_s)
            parent_idx = int(current_start // parent_duration_s)
            parent_id = f"{vid}_p{parent_idx:02d}"

            chunks.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_index=idx,
                    start_s=round(current_start, 3),
                    end_s=round(current_end, 3),
                    duration_s=round(dur, 3),
                    start_ms=int(round(current_start * 1000)),
                    end_ms=int(round(current_end * 1000)),
                    boundary_type="fixed_window",
                    boundary_confidence=0.5,
                    metadata={"step_s": step_s}
                )
            )
            idx += 1
            current_start += step_s
            # Break if remaining segment is negligible
            if video_duration_s - current_start < 0.2:
                break

        # Assemble Parent Segments
        parents = self._assemble_parent_segments(chunks, vid, parent_duration_s, video_duration_s)

        elapsed = time.perf_counter() - t0
        throughput = (video_duration_s / elapsed) if elapsed > 0 else 9999.0
        avg_dur = float(np.mean([c.duration_s for c in chunks])) if chunks else 0.0

        return ChunkManifest(
            video_id=vid,
            video_path=video_path,
            total_duration_s=video_duration_s,
            strategy="fixed_duration_baseline",
            policy_name="fixed_duration",
            parent_segments=parents,
            chunks=chunks,
            num_chunks=len(chunks),
            num_parents=len(parents),
            average_chunk_duration_s=round(avg_dur, 3),
            overlap_ratio=self.overlap_ratio,
            processing_time_s=round(elapsed, 4),
            throughput_factor=round(throughput, 2),
        )

    def _assemble_parent_segments(
        self,
        chunks: List[ChildChunk],
        video_id: str,
        parent_duration_s: float,
        total_duration_s: float
    ) -> List[ParentSegment]:
        if not chunks:
            return []
        parents: List[ParentSegment] = []
        p_map: Dict[str, List[ChildChunk]] = {}
        for c in chunks:
            p_map.setdefault(c.parent_id, []).append(c)

        for p_idx, (pid, p_chunks) in enumerate(sorted(p_map.items(), key=lambda x: x[1][0].start_s)):
            p_start = p_chunks[0].start_s
            p_end = p_chunks[-1].end_s
            parents.append(
                ParentSegment(
                    segment_id=pid,
                    segment_index=p_idx,
                    start_s=round(p_start, 3),
                    end_s=round(p_end, 3),
                    duration_s=round(p_end - p_start, 3),
                    start_ms=int(round(p_start * 1000)),
                    end_ms=int(round(p_end * 1000)),
                    child_chunk_ids=[c.chunk_id for c in p_chunks],
                    topic_label=f"Fixed Section {p_idx + 1}"
                )
            )
        return parents


class BlindSceneCutChunker:
    """
    Naive baseline that splits strictly at every detected scene cut,
    without enforcing minimum/maximum duration bounds or overlap constraints.
    """

    def __init__(self, scene_threshold: float = 0.35):
        self.detector = MultiSignalDetector(scene_threshold=scene_threshold)

    def chunk(
        self,
        video_path: str,
        video_duration_s: float,
        video_id: Optional[str] = None,
        parent_duration_s: float = 180.0,
    ) -> ChunkManifest:
        """Executes unconstrained scene-cut splitting."""
        t0 = time.perf_counter()
        vid = video_id or (video_path.split("/")[-1].split("\\")[-1].split(".")[0])
        candidates = self.detector.detect_signals(video_path, video_duration_s=video_duration_s)

        # Cut timestamps including 0.0 and video_duration_s
        cuts = [0.0] + [c.timestamp_s for c in candidates if 0.0 < c.timestamp_s < video_duration_s] + [video_duration_s]
        cuts = sorted(list(set(cuts)))

        chunks: List[ChildChunk] = []
        for idx in range(len(cuts) - 1):
            c_start = cuts[idx]
            c_end = cuts[idx + 1]
            dur = c_end - c_start
            chunk_id = f"{vid}_c{idx:03d}"
            parent_idx = int(c_start // parent_duration_s)
            parent_id = f"{vid}_p{parent_idx:02d}"

            chunks.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    chunk_index=idx,
                    start_s=round(c_start, 3),
                    end_s=round(c_end, 3),
                    duration_s=round(dur, 3),
                    start_ms=int(round(c_start * 1000)),
                    end_ms=int(round(c_end * 1000)),
                    boundary_type="scene_cut",
                    boundary_confidence=0.8,
                    metadata={"unconstrained_scene": True}
                )
            )

        elapsed = time.perf_counter() - t0
        throughput = (video_duration_s / elapsed) if elapsed > 0 else 9999.0
        avg_dur = float(np.mean([c.duration_s for c in chunks])) if chunks else 0.0

        parents = FixedDurationChunker()._assemble_parent_segments(
            chunks, vid, parent_duration_s, video_duration_s
        )

        return ChunkManifest(
            video_id=vid,
            video_path=video_path,
            total_duration_s=video_duration_s,
            strategy="blind_scenecut_baseline",
            policy_name="blind_scenecut",
            parent_segments=parents,
            chunks=chunks,
            num_chunks=len(chunks),
            num_parents=len(parents),
            average_chunk_duration_s=round(avg_dur, 3),
            overlap_ratio=0.0,
            processing_time_s=round(elapsed, 4),
            throughput_factor=round(throughput, 2),
        )


import numpy as np
