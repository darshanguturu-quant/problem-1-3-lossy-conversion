"""
User Intent Classifier Package
Implements Stage 1 TF-IDF fast gate, Stage 2 Sentence Embeddings, and Cascaded Hybrid Router.
"""

from .features import normalize_query, GrammaticalCueExtractor, HybridFeaturePipeline
from .stage1 import Stage1IntentClassifier, TemperatureScaler, INTENT_CLASSES
from .stage2 import Stage2IntentClassifier, SentenceEmbeddingExtractor
from .cascade import CascadedIntentClassifier, IntentCascadeResult
from .baselines import IntentBenchmarkRunner, IntentMetricReport
from .data import build_intent_dataset, label_videomme_question

__all__ = [
    "normalize_query",
    "GrammaticalCueExtractor",
    "HybridFeaturePipeline",
    "Stage1IntentClassifier",
    "TemperatureScaler",
    "INTENT_CLASSES",
    "Stage2IntentClassifier",
    "SentenceEmbeddingExtractor",
    "CascadedIntentClassifier",
    "IntentCascadeResult",
    "IntentBenchmarkRunner",
    "IntentMetricReport",
    "build_intent_dataset",
    "label_videomme_question",
]
