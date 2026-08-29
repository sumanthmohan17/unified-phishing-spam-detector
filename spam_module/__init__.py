"""
Email Spam Classification Module
================================
Module 2: Email Classification System (Legitimate, Spam, Phishing, Malware)
with automated URL extraction and live routing into Module 1 (Phishing Detection).
"""

from .classifier import (
    classify_email,
    extract_spam_features,
    generate_synthetic_spam_data,
    train_spam_ensemble,
)
from .preprocessing import preprocess_email_text
from .url_extraction import extract_urls_from_email

__all__ = [
    "preprocess_email_text",
    "extract_urls_from_email",
    "extract_spam_features",
    "train_spam_ensemble",
    "classify_email",
    "generate_synthetic_spam_data",
]
