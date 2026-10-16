"""
Data models and schemas for Task 5 Retrieval Benchmark Builder & Evaluator.
Defines stratified benchmark query records, ground-truth evidence spans,
hard negatives, ranking results, and evaluation scorecards.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import List, Dict, Any, Optional, Literal, Tuple


IntentType = Literal["whole_video", "frame_level", "clip_level"]
DifficultyType = Literal["easy", "medium", "hard"]
ModalityType = Literal["visual", "speech_audio", "ocr_text", "multimodal"]
VerificationStatus = Literal["verified_gold", "evidence_grounded", "flagged_uncertain"]


@dataclass
class TemporalSpan:
    """Represents a ground-truth evidence interval in seconds."""
    start_s: float
    end_s: float
    timestamp_s: Optional[float] = None  # Exact instant for frame_level

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class BenchmarkQuery:
    """An evidence-grounded evaluation query with validated target answers and distractors."""
    query_id: str
    dataset: str  # "dataset_a", "dataset_b", "dataset_c", "videomme"
    video_id: str
    query: str
    query_intent: IntentType
    target_granularity: str  # "summary_index", "frame_index", "clip_index"
    temporal_anchor: TemporalSpan
    difficulty: DifficultyType
    evidence_modality: ModalityType
    expected_answer: str
    options: Optional[List[str]] = None  # For multiple-choice QA
    hard_negatives: List[str] = field(default_factory=list)  # Distractor video/chunk IDs
    verification_status: VerificationStatus = "evidence_grounded"
    uncertainty_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["temporal_anchor"] = self.temporal_anchor.to_dict()
        return d


@dataclass
class BenchmarkSplit:
    """A partition of queries strictly grouped by video_id."""
    split_name: str  # "train", "val", "test"
    queries: List[BenchmarkQuery] = field(default_factory=list)
    video_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "num_queries": len(self.queries),
            "num_videos": len(self.video_ids),
            "video_ids": self.video_ids,
            "queries": [q.to_dict() for q in self.queries],
        }


@dataclass
class BenchmarkManifest:
    """Complete multi-dataset retrieval benchmark manifest."""
    benchmark_id: str
    version: str
    created_at: str
    random_seed: int
    datasets_included: List[str]
    total_queries: int
    total_videos: int
    splits: Dict[str, BenchmarkSplit] = field(default_factory=dict)
    stratification_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "version": self.version,
            "created_at": self.created_at,
            "random_seed": self.random_seed,
            "datasets_included": self.datasets_included,
            "total_queries": self.total_queries,
            "total_videos": self.total_videos,
            "stratification_summary": self.stratification_summary,
            "metadata": self.metadata,
            "splits": {k: v.to_dict() for k, v in self.splits.items()},
        }

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: str) -> "BenchmarkManifest":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        splits = {}
        for sname, sdata in data.get("splits", {}).items():
            queries = []
            for qd in sdata.get("queries", []):
                span_data = qd["temporal_anchor"]
                tspan = TemporalSpan(
                    start_s=span_data["start_s"],
                    end_s=span_data["end_s"],
                    timestamp_s=span_data.get("timestamp_s")
                )
                q = BenchmarkQuery(
                    query_id=qd["query_id"],
                    dataset=qd["dataset"],
                    video_id=qd["video_id"],
                    query=qd["query"],
                    query_intent=qd["query_intent"],
                    target_granularity=qd["target_granularity"],
                    temporal_anchor=tspan,
                    difficulty=qd["difficulty"],
                    evidence_modality=qd["evidence_modality"],
                    expected_answer=qd["expected_answer"],
                    options=qd.get("options"),
                    hard_negatives=qd.get("hard_negatives", []),
                    verification_status=qd.get("verification_status", "evidence_grounded"),
                    uncertainty_reason=qd.get("uncertainty_reason"),
                    metadata=qd.get("metadata", {}),
                )
                queries.append(q)
            splits[sname] = BenchmarkSplit(
                split_name=sname,
                queries=queries,
                video_ids=sdata.get("video_ids", [])
            )

        return cls(
            benchmark_id=data["benchmark_id"],
            version=data["version"],
            created_at=data["created_at"],
            random_seed=data.get("random_seed", 42),
            datasets_included=data.get("datasets_included", []),
            total_queries=data.get("total_queries", 0),
            total_videos=data.get("total_videos", 0),
            splits=splits,
            stratification_summary=data.get("stratification_summary", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueryRetrievalPrediction:
    """Retrieval prediction for an individual query."""
    query_id: str
    ranked_video_ids: List[str]  # Ranked list of retrieved video IDs
    ranked_chunk_spans: Optional[List[Tuple[float, float]]] = None  # Ranked time intervals (start_s, end_s)
    predicted_answer: Optional[str] = None
    retrieval_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingMetrics:
    """Standardized retrieval ranking metrics."""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mean_reciprocal_rank: float  # MRR
    ndcg_at_10: float
    context_precision: float
    evidence_temporal_iou: float  # t-IoU
    predictive_answer_accuracy: float
    retrieval_answer_correlation: float  # R^2 / alignment
    p50_latency_ms: float
    p95_latency_ms: float
    num_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationScorecard:
    """Complete evaluation report covering in-domain, cross-domain transfer, and mixed-corpus."""
    evaluator_name: str
    evaluation_mode: str  # "in_domain", "cross_domain_transfer", "mixed_corpus"
    overall_metrics: RankingMetrics
    per_dataset_metrics: Dict[str, RankingMetrics] = field(default_factory=dict)
    per_intent_metrics: Dict[str, RankingMetrics] = field(default_factory=dict)
    per_difficulty_metrics: Dict[str, RankingMetrics] = field(default_factory=dict)
    cross_dataset_transfer_matrix: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "evaluation_mode": self.evaluation_mode,
            "overall_metrics": self.overall_metrics.to_dict(),
            "per_dataset_metrics": {k: v.to_dict() for k, v in self.per_dataset_metrics.items()},
            "per_intent_metrics": {k: v.to_dict() for k, v in self.per_intent_metrics.items()},
            "per_difficulty_metrics": {k: v.to_dict() for k, v in self.per_difficulty_metrics.items()},
            "cross_dataset_transfer_matrix": self.cross_dataset_transfer_matrix,
            "metadata": self.metadata,
        }

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
