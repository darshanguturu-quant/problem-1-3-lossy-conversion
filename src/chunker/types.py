"""
Data models and manifests for Task 4 Intelligent Video Chunking.
Defines two-level parent-child hierarchical segmentation structures and boundary candidate objects.
"""

from dataclasses import dataclass, field, asdict
import json
import os
import uuid
from typing import List, Dict, Any, Optional


@dataclass
class BoundaryCandidate:
    """A detected boundary transition candidate from one or more modalities."""
    timestamp_s: float
    signal_type: str  # "scene_cut", "speech_pause", "motion_onset", "ocr_slide", "fused"
    confidence: float
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChildChunk:
    """Fine-grained temporal chunk corresponding to a coherent visual/speech scene."""
    chunk_id: str
    parent_id: str
    chunk_index: int
    start_s: float
    end_s: float
    duration_s: float
    start_ms: int
    end_ms: int
    boundary_type: str  # "scene_cut", "speech_pause", "motion_onset", "ocr_slide", "clamped_max", "window_fallback", "fixed_window"
    boundary_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParentSegment:
    """Coarse-grained hierarchical segment grouping contiguous child chunks into topical sections."""
    segment_id: str
    segment_index: int
    start_s: float
    end_s: float
    duration_s: float
    start_ms: int
    end_ms: int
    child_chunk_ids: List[str] = field(default_factory=list)
    topic_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkManifest:
    """Complete serialized chunking manifest for a video."""
    video_id: str
    video_path: str
    total_duration_s: float
    strategy: str  # "intelligent_content_aware", "fixed_duration_baseline", "blind_scenecut_baseline"
    policy_name: str
    parent_segments: List[ParentSegment] = field(default_factory=list)
    chunks: List[ChildChunk] = field(default_factory=list)
    num_chunks: int = 0
    num_parents: int = 0
    average_chunk_duration_s: float = 0.0
    overlap_ratio: float = 0.0
    processing_time_s: float = 0.0
    throughput_factor: float = 0.0  # media_seconds / wall_seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "total_duration_s": round(self.total_duration_s, 3),
            "strategy": self.strategy,
            "policy_name": self.policy_name,
            "num_parents": self.num_parents,
            "num_chunks": self.num_chunks,
            "average_chunk_duration_s": round(self.average_chunk_duration_s, 3),
            "overlap_ratio": round(self.overlap_ratio, 4),
            "processing_time_s": round(self.processing_time_s, 4),
            "throughput_factor": round(self.throughput_factor, 2),
            "metadata": self.metadata,
            "parent_segments": [p.to_dict() for p in self.parent_segments],
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def save(self, output_path: str) -> None:
        """Saves manifest to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "ChunkManifest":
        """Loads manifest from a JSON file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        parents = [ParentSegment(**p) for p in data.get("parent_segments", [])]
        chunks = [ChildChunk(**c) for c in data.get("chunks", [])]
        return cls(
            video_id=data["video_id"],
            video_path=data["video_path"],
            total_duration_s=data["total_duration_s"],
            strategy=data["strategy"],
            policy_name=data["policy_name"],
            parent_segments=parents,
            chunks=chunks,
            num_chunks=data.get("num_chunks", len(chunks)),
            num_parents=data.get("num_parents", len(parents)),
            average_chunk_duration_s=data.get("average_chunk_duration_s", 0.0),
            overlap_ratio=data.get("overlap_ratio", 0.0),
            processing_time_s=data.get("processing_time_s", 0.0),
            throughput_factor=data.get("throughput_factor", 0.0),
            metadata=data.get("metadata", {}),
        )
