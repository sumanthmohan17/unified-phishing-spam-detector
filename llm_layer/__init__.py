"""
LLM Explainability Layer Module
===============================
Module 3: Plain-English Threat Explanations & Action Recommendations via Groq API.
"""

from .explainer import (
    explain_phishing_detection,
    explain_spam_detection,
)

__all__ = [
    "explain_phishing_detection",
    "explain_spam_detection",
]
