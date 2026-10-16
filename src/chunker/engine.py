"""
Intelligent Video Chunking Engine (Task 4).
Content-aware, adaptive temporal segmentation with hierarchical parent-child schemas.
Guarantees O(D * fs) linear execution, 5%-15% overlap compliance, and robust error handling.
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from src.chunker.types import (
    BoundaryCandidate,
    ChildChunk,
    ParentSegment,
    ChunkManifest,
)
from src.chunker.policies import ChunkingPolicy, get_policy
from src.chunker.signals import MultiSignalDetector, detect_signals_fast

logger = logging.getLogger("aintropy-chunker.engine")


class IntelligentChunker:
    """
    Content-aware video chunking engine.
    Finds natural semantic/visual/speech boundaries within genre-specific duration windows.
    """

    def __init__(
        self,
        policy: Optional[ChunkingPolicy] = None,
        sample_fps: float = 3.0,
        enable_ocr: bool = True,
    ):
        self.policy = policy or get_policy("general")
        self.detector = MultiSignalDetector(
            sample_fps=sample_fps,
            scene_threshold=self.policy.scene_threshold,
            enable_ocr_slide=enable_ocr and self.policy.ocr_refinement,
        )

    def chunk_video(
        self,
        video_path: str,
        video_type: Optional[str] = None,
        video_duration_s: Optional[float] = None,
        custom_cues: Optional[List[Dict[str, Any]]] = None,
        policy: Optional[ChunkingPolicy] = None,
        video_id: Optional[str] = None,
    ) -> ChunkManifest:
        """
        Executes content-aware adaptive chunking with parent-child hierarchical segmentation.
        """
        t0 = time.perf_counter()
        active_policy = policy or (get_policy(video_type) if video_type else self.policy)
        vid = video_id or self._extract_video_id(video_path)

        # 1. Probe duration if not supplied
        duration_s = video_duration_s
        if duration_s is None or duration_s <= 0:
            duration_s = self._probe_duration(video_path)

        # Handle corrupt or empty media gracefully
        if duration_s <= 0.0:
            logger.warning("Video duration is 0 or unreadable for '%s'. Emitting empty manifest.", video_path)
            return ChunkManifest(
                video_id=vid,
                video_path=video_path,
                total_duration_s=0.0,
                strategy="intelligent_content_aware",
                policy_name=active_policy.name,
                num_chunks=0,
                num_parents=0,
                metadata={"error": "unreadable_or_zero_duration"}
            )

        # 2. Extract Multimodal Boundary Candidates (Linear O(D * fs) single pass)
        candidates = self.detector.detect_signals(
            video_path=video_path,
            video_duration_s=duration_s,
            custom_cues=custom_cues,
            ocr_refinement=active_policy.ocr_refinement,
        )

        # 3. Forward-Pointer Adaptive Windowing
        chunks = self._partition_into_child_chunks(
            candidates=candidates,
            total_duration_s=duration_s,
            policy=active_policy,
            video_id=vid,
        )

        # 4. Hierarchical Parent Segment Synthesis
        parents = self._synthesize_parent_segments(
            chunks=chunks,
            parent_target_duration_s=active_policy.parent_target_duration_s,
            total_duration_s=duration_s,
            video_id=vid,
        )

        elapsed = time.perf_counter() - t0
        throughput = (duration_s / elapsed) if elapsed > 0 else 9999.0
        avg_dur = float(np.mean([c.duration_s for c in chunks])) if chunks else 0.0

        # Calculate actual measured overlap ratio
        measured_overlap = self._compute_effective_overlap(chunks)

        return ChunkManifest(
            video_id=vid,
            video_path=video_path,
            total_duration_s=round(duration_s, 3),
            strategy="intelligent_content_aware",
            policy_name=active_policy.name,
            parent_segments=parents,
            chunks=chunks,
            num_chunks=len(chunks),
            num_parents=len(parents),
            average_chunk_duration_s=round(avg_dur, 3),
            overlap_ratio=round(measured_overlap, 4),
            processing_time_s=round(elapsed, 4),
            throughput_factor=round(throughput, 2),
            metadata={
                "target_duration_s": active_policy.target_duration_s,
                "min_duration_s": active_policy.min_duration_s,
                "max_duration_s": active_policy.max_duration_s,
                "configured_overlap": active_policy.overlap_ratio,
                "candidates_detected": len(candidates),
            }
        )

    def _partition_into_child_chunks(
        self,
        candidates: List[BoundaryCandidate],
        total_duration_s: float,
        policy: ChunkingPolicy,
        video_id: str,
    ) -> List[ChildChunk]:
        """
        Executes forward-pointer boundary snapping matching candidate transitions to duration limits.
        """
        chunks: List[ChildChunk] = []
        cand_ts = [c.timestamp_s for c in candidates]

        current_start = 0.0
        idx = 0

        while current_start < total_duration_s:
            # If the remaining duration is less than or close to max duration, handle end
            remaining = total_duration_s - current_start
            if remaining <= policy.max_duration_s:
                end_s = total_duration_s
                b_type = "video_end"
                b_conf = 1.0
            else:
                target_end = current_start + policy.target_duration_s
                min_end = current_start + policy.min_duration_s
                max_end = min(total_duration_s, current_start + policy.max_duration_s)

                # Preferred candidate search window around target end
                pref_min = max(min_end, target_end - policy.tolerance_s)
                pref_max = min(max_end, target_end + policy.tolerance_s)

                # 1. Search for candidates inside preferred window
                cands_in_pref = [
                    c for c in candidates
                    if pref_min <= c.timestamp_s <= pref_max
                ]

                if cands_in_pref:
                    # Select candidate maximizing composite score (confidence penalized by distance from target)
                    best_cand = max(
                        cands_in_pref,
                        key=lambda c: (c.confidence - 0.03 * abs(c.timestamp_s - target_end))
                    )
                    end_s = best_cand.timestamp_s
                    b_type = best_cand.signal_type
                    b_conf = best_cand.confidence
                else:
                    # 2. Search anywhere inside allowed min/max window
                    cands_in_allowed = [
                        c for c in candidates
                        if min_end <= c.timestamp_s <= max_end
                    ]
                    if cands_in_allowed:
                        best_cand = min(cands_in_allowed, key=lambda c: abs(c.timestamp_s - target_end))
                        end_s = best_cand.timestamp_s
                        b_type = best_cand.signal_type
                        b_conf = best_cand.confidence
                    else:
                        # 3. Fallback: clamp to target duration
                        end_s = min(max_end, target_end)
                        b_type = "window_fallback"
                        b_conf = 0.50

                # Check for orphan fragment protection at end of video
                if (total_duration_s - end_s) < policy.min_duration_s:
                    if (total_duration_s - current_start) <= (policy.max_duration_s * 1.25):
                        end_s = total_duration_s
                        b_type = "orphan_absorbed"

            dur = end_s - current_start
            chunk_id = f"{video_id}_c{idx:03d}"
            # Temporary parent_id (will be reassigned during parent synthesis)
            temp_parent_id = f"{video_id}_p00"

            chunks.append(
                ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=temp_parent_id,
                    chunk_index=idx,
                    start_s=round(current_start, 3),
                    end_s=round(end_s, 3),
                    duration_s=round(dur, 3),
                    start_ms=int(round(current_start * 1000)),
                    end_ms=int(round(end_s * 1000)),
                    boundary_type=b_type,
                    boundary_confidence=round(b_conf, 3),
                    metadata={
                        "target_duration_s": policy.target_duration_s,
                        "tolerance_s": policy.tolerance_s,
                    }
                )
            )

            if end_s >= total_duration_s:
                break

            # Calculate next start using configured sliding overlap
            overlap_s = policy.overlap_ratio * dur
            next_start = end_s - overlap_s

            # Ensure forward progress
            if next_start <= current_start:
                next_start = current_start + (dur * 0.5)

            current_start = next_start
            idx += 1

        return chunks

    def _synthesize_parent_segments(
        self,
        chunks: List[ChildChunk],
        parent_target_duration_s: float,
        total_duration_s: float,
        video_id: str,
    ) -> List[ParentSegment]:
        """
        Groups contiguous child chunks into parent segments representing coarse topic sections.
        Ensures strict parent-child containment without orphan child chunks.
        """
        if not chunks:
            return []

        parents: List[ParentSegment] = []
        p_idx = 0
        current_child_ids: List[str] = []
        p_start_s = chunks[0].start_s

        for i, chunk in enumerate(chunks):
            current_child_ids.append(chunk.chunk_id)
            accumulated_dur = chunk.end_s - p_start_s
            is_last = (i == len(chunks) - 1)

            # Close parent segment when target duration reached or at end of media
            if accumulated_dur >= parent_target_duration_s or is_last:
                pid = f"{video_id}_p{p_idx:02d}"
                p_end_s = chunk.end_s
                parents.append(
                    ParentSegment(
                        segment_id=pid,
                        segment_index=p_idx,
                        start_s=round(p_start_s, 3),
                        end_s=round(p_end_s, 3),
                        duration_s=round(p_end_s - p_start_s, 3),
                        start_ms=int(round(p_start_s * 1000)),
                        end_ms=int(round(p_end_s * 1000)),
                        child_chunk_ids=list(current_child_ids),
                        topic_label=f"Topic Section {p_idx + 1}"
                    )
                )

                # Update child chunk parent_id pointers
                for cid in current_child_ids:
                    for c in chunks:
                        if c.chunk_id == cid:
                            c.parent_id = pid

                p_idx += 1
                current_child_ids = []
                if not is_last and (i + 1 < len(chunks)):
                    p_start_s = chunks[i + 1].start_s

        return parents

    def _compute_effective_overlap(self, chunks: List[ChildChunk]) -> float:
        """Computes the average overlap ratio between consecutive child chunks."""
        if len(chunks) <= 1:
            return 0.0

        overlaps: List[float] = []
        for i in range(len(chunks) - 1):
            c_curr = chunks[i]
            c_next = chunks[i + 1]
            overlap_dur = max(0.0, c_curr.end_s - c_next.start_s)
            ratio = overlap_dur / c_curr.duration_s if c_curr.duration_s > 0 else 0.0
            overlaps.append(ratio)

        return float(np.mean(overlaps)) if overlaps else 0.0

    def _extract_video_id(self, video_path: str) -> str:
        base = os.path.basename(video_path)
        name, _ = os.path.splitext(base)
        return name or "video"

    def _probe_duration(self, video_path: str) -> float:
        """Probes video duration via OpenCV."""
        if not os.path.exists(video_path):
            return 0.0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and count > 0:
            return count / fps
        return 0.0


def chunk_video(
    video_path: str,
    video_type: Optional[str] = "general",
    output_path: Optional[str] = None,
    target_duration_s: Optional[float] = None,
    min_duration_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    overlap_ratio: Optional[float] = None,
    custom_cues: Optional[List[Dict[str, Any]]] = None,
    policy: Optional[ChunkingPolicy] = None,
) -> ChunkManifest:
    """
    High-level entrypoint to segment a video and optionally save the JSON manifest.
    """
    active_policy = policy or get_policy(
        video_type=video_type,
        target_duration_s=target_duration_s,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        overlap_ratio=overlap_ratio,
    )
    chunker = IntelligentChunker(policy=active_policy)
    manifest = chunker.chunk_video(
        video_path=video_path,
        video_type=video_type,
        custom_cues=custom_cues,
        policy=active_policy,
    )

    if output_path:
        manifest.save(output_path)
        logger.info("Saved chunk manifest to '%s'", output_path)

    return manifest
