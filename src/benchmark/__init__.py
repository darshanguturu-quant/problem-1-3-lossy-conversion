"""
AIntropy Multimodal Video Retrieval Benchmark Builder & Evaluator (Task 5).
Generates evidence-grounded query sets across VideoMME and production datasets A, B, and C,
evaluates ranking, predictive answering, and 3x3 cross-dataset domain transfer.
"""

from src.benchmark.types import (
    TemporalSpan,
    BenchmarkQuery,
    BenchmarkSplit,
    BenchmarkManifest,
    QueryRetrievalPrediction,
    RankingMetrics,
    EvaluationScorecard,
)
from src.benchmark.builder import RetrievalBenchmarkBuilder
from src.benchmark.evaluator import RetrievalEvaluator
from src.benchmark.search import HybridRetrievalEngine, VideoDocumentIndex
from src.benchmark.transfer import CrossDatasetTransferEvaluator

__all__ = [
    "TemporalSpan",
    "BenchmarkQuery",
    "BenchmarkSplit",
    "BenchmarkManifest",
    "QueryRetrievalPrediction",
    "RankingMetrics",
    "EvaluationScorecard",
    "RetrievalBenchmarkBuilder",
    "RetrievalEvaluator",
    "HybridRetrievalEngine",
    "VideoDocumentIndex",
    "CrossDatasetTransferEvaluator",
]
