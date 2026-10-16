"""
AIntropy Intelligent Video Chunking Engine (Task 4)
Content-aware adaptive video segmentation with hierarchical parent-child structures.
"""

from src.chunker.types import (
    BoundaryCandidate,
    ChildChunk,
    ParentSegment,
    ChunkManifest,
)
from src.chunker.policies import (
    ChunkingPolicy,
    get_policy,
    list_policies,
    CATEGORY_POLICIES,
)
from src.chunker.signals import (
    MultiSignalDetector,
    detect_signals_fast,
)
from src.chunker.baselines import (
    FixedDurationChunker,
    BlindSceneCutChunker,
)
from src.chunker.engine import (
    IntelligentChunker,
    chunk_video,
)
from src.chunker.benchmarks import (
    ChunkerBenchmarkRunner,
    BenchmarkScorecard,
)

__all__ = [
    "BoundaryCandidate",
    "ChildChunk",
    "ParentSegment",
    "ChunkManifest",
    "ChunkingPolicy",
    "get_policy",
    "list_policies",
    "CATEGORY_POLICIES",
    "MultiSignalDetector",
    "detect_signals_fast",
    "FixedDurationChunker",
    "BlindSceneCutChunker",
    "IntelligentChunker",
    "chunk_video",
    "ChunkerBenchmarkRunner",
    "BenchmarkScorecard",
]
