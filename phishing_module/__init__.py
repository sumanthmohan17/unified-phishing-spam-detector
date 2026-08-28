"""
Phishing Detection Module
=========================
Module 1: Multi-Signal Phishing Detection System.
Provides feature extractors and ensemble classifiers for:
1. URL Structural Features (feature_extraction.py)
2. Visual Brand Similarity (visual_similarity.py)
3. NLP Page Content Analysis (content_features.py)
4. Multi-Signal Stacking Classifier (classifier.py)
"""

from .classifier import (
    build_feature_vector,
    generate_synthetic_training_data,
    predict,
    train_ensemble,
)
from .content_features import (
    extract_content_features,
    extract_content_features_batch,
)
from .feature_extraction import (
    extract_url_features,
    extract_url_features_batch,
)
from .visual_similarity import (
    build_reference_library,
    compare_screenshot,
)

__all__ = [
    "extract_url_features",
    "extract_url_features_batch",
    "compare_screenshot",
    "build_reference_library",
    "extract_content_features",
    "extract_content_features_batch",
    "build_feature_vector",
    "train_ensemble",
    "predict",
    "generate_synthetic_training_data",
]
