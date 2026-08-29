"""
Threat Intelligence Cross-Validation Module
===========================================
Module 4: Cross-validation of URLs against VirusTotal API v3 and Google Safe Browsing API v4
with deterministic verdict escalation overriding ML classification.
"""

from .cross_validator import (
    cross_validate_url,
    enrich_detection_with_threat_intel,
)
from .safe_browsing_check import check_url_safe_browsing
from .virustotal_check import check_url_virustotal

__all__ = [
    "check_url_virustotal",
    "check_url_safe_browsing",
    "cross_validate_url",
    "enrich_detection_with_threat_intel",
]
