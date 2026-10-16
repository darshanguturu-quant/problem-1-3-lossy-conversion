"""
Video Type Classifier Package
Implements Stage 1 fast gate, Stage 2 CLIP refinement, adaptive frame sampling, and baselines.
"""

from .stage1 import Stage1Classifier, Stage1Features
from .stage2 import Stage2Classifier, AdaptiveFrameSampler, CLIPFeatureExtractor
from .cascade import CascadedClassifier, CascadeResult, FeatureCache
from .baselines import MobileNetV3Classifier, BenchmarkRunner

__all__ = [
    "Stage1Classifier",
    "Stage1Features",
    "Stage2Classifier",
    "AdaptiveFrameSampler",
    "CLIPFeatureExtractor",
    "CascadedClassifier",
    "CascadeResult",
    "FeatureCache",
    "MobileNetV3Classifier",
    "BenchmarkRunner",
]
