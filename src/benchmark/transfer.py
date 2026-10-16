"""
Cross-Dataset Generalization & 3x3 Transfer Matrix Evaluator (Task 5).
Evaluates In-Domain retrieval, Zero-Shot Cross-Domain Transfer (A+B -> C),
and Mixed-Corpus retrieval with global distractors across heterogeneous domains.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from src.benchmark.types import (
    BenchmarkQuery,
    BenchmarkManifest,
    QueryRetrievalPrediction,
    EvaluationScorecard,
    RankingMetrics,
)
from src.benchmark.search import HybridRetrievalEngine
from src.benchmark.evaluator import RetrievalEvaluator

logger = logging.getLogger("aintropy-benchmark.transfer")


class CrossDatasetTransferEvaluator:
    """
    Evaluates cross-dataset domain transfer gaps and mixed-corpus distractor robustness.
    """

    def __init__(self, evaluator: Optional[RetrievalEvaluator] = None):
        self.evaluator = evaluator or RetrievalEvaluator()

    def evaluate_3x3_transfer_matrix(
        self,
        manifest: BenchmarkManifest,
        split_name: str = "test",
        top_k: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes 3x3 transfer matrix (Recall@10) across Dataset A, Dataset B, and Dataset C:
        Rows: Training / Tuning Source Dataset
        Cols: Evaluation Target Dataset
        """
        target_datasets = ["dataset_a", "dataset_b", "dataset_c"]
        split = manifest.splits.get(split_name)
        if not split:
            raise ValueError(f"Split '{split_name}' not found in manifest!")

        queries_by_dataset: Dict[str, List[BenchmarkQuery]] = {}
        for ds in target_datasets:
            queries_by_dataset[ds] = [q for q in split.queries if q.dataset == ds]

        # Pre-build index for each dataset once to avoid redundant re-encoding
        dataset_engines: Dict[str, HybridRetrievalEngine] = {}
        for ds in target_datasets:
            ds_queries = queries_by_dataset.get(ds, [])
            if ds_queries:
                eng = HybridRetrievalEngine(use_dense_embeddings=True)
                eng.build_index(ds_queries)
                dataset_engines[ds] = eng

        transfer_matrix: Dict[str, Dict[str, float]] = {}

        for src_ds in target_datasets:
            transfer_matrix[src_ds] = {}
            for eval_ds in target_datasets:
                eval_queries = queries_by_dataset.get(eval_ds, [])
                if not eval_queries:
                    transfer_matrix[src_ds][eval_ds] = 0.0
                    continue

                if src_ds == eval_ds:
                    # In-Domain evaluation: retrieve within source corpus
                    eng = dataset_engines.get(src_ds)
                    if eng:
                        preds = [eng.retrieve(q, top_k=top_k) for q in eval_queries]
                        metrics = self.evaluator._compute_metrics_subset(eval_queries, preds)
                        transfer_matrix[src_ds][eval_ds] = metrics.recall_at_10
                    else:
                        transfer_matrix[src_ds][eval_ds] = 0.0
                else:
                    # Cross-Domain Zero-Shot evaluation:
                    # Index the target corpus using the source-trained lexical vectorizer (zero-shot vocabulary transfer)
                    src_eng = dataset_engines.get(src_ds)
                    if src_eng:
                        transfer_eng = HybridRetrievalEngine(use_dense_embeddings=True)
                        transfer_eng.tfidf_vectorizer = src_eng.tfidf_vectorizer
                        transfer_eng.build_index(eval_queries, fit_vectorizer=False)
                        preds = [transfer_eng.retrieve(q, top_k=top_k) for q in eval_queries]
                        metrics = self.evaluator._compute_metrics_subset(eval_queries, preds)
                        transfer_matrix[src_ds][eval_ds] = metrics.recall_at_10
                    else:
                        transfer_matrix[src_ds][eval_ds] = 0.0

        return transfer_matrix

    def evaluate_mixed_corpus(
        self,
        manifest: BenchmarkManifest,
        split_name: str = "test",
        top_k: int = 10,
    ) -> EvaluationScorecard:
        """
        Evaluates mixed-corpus retrieval where all datasets (A, B, C, VideoMME) are pooled
        into a single large index, creating challenging cross-domain distractor conditions.
        """
        split = manifest.splits.get(split_name)
        if not split:
            raise ValueError(f"Split '{split_name}' not found in manifest!")

        test_queries = split.queries
        logger.info("Building global mixed-corpus index across %d test queries...", len(test_queries))

        engine = HybridRetrievalEngine(use_dense_embeddings=True)
        engine.build_index(test_queries)

        preds = [engine.retrieve(q, top_k=top_k) for q in test_queries]

        transfer_matrix = self.evaluate_3x3_transfer_matrix(manifest, split_name=split_name, top_k=top_k)

        scorecard = self.evaluator.evaluate_predictions(
            queries=test_queries,
            predictions=preds,
            evaluator_name="mixed_corpus_hybrid_rrf",
            evaluation_mode="mixed_corpus_with_global_distractors",
            transfer_matrix=transfer_matrix,
        )
        return scorecard
