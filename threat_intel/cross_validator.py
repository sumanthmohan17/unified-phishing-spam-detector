"""
Threat Intelligence Cross-Validation & Escalation Module
========================================================
Module 4: Cross-validation of URLs against VirusTotal and Google Safe Browsing
with strict threat intelligence escalation overriding ML model verdicts
as described in Section 3.2.1 and Section VIII-D of the project report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .safe_browsing_check import check_url_safe_browsing
from .virustotal_check import check_url_virustotal


def cross_validate_url(
    url: str,
    enable_vt_rate_limit: bool = True,
) -> Dict[str, Any]:
    """
    Cross-validate a URL against both VirusTotal (v3) and Google Safe Browsing (v4).

    Parameters
    ----------
    url : str
        The URL to cross-validate.
    enable_vt_rate_limit : bool, default=True
        Whether to enforce rate-limiting on VirusTotal calls.

    Returns
    -------
    dict
        {
            "url": str,
            "is_confirmed_malicious": bool,
            "virustotal_result": dict,
            "safe_browsing_result": dict,
            "escalation_reason": Optional[str],
        }
    """
    vt_res = check_url_virustotal(url, enable_rate_limit=enable_vt_rate_limit)
    sb_res = check_url_safe_browsing(url)

    is_confirmed_malicious = bool(vt_res.get("flagged") or sb_res.get("flagged"))

    reasons: List[str] = []
    if vt_res.get("flagged"):
        reasons.append(
            f"VirusTotal ({vt_res.get('malicious_count')}/{vt_res.get('total_engines')} engines)"
        )
    if sb_res.get("flagged"):
        threats_str = ", ".join(sb_res.get("threat_types", [])) or "THREAT"
        reasons.append(f"Google Safe Browsing ({threats_str})")

    if reasons:
        escalation_reason = f"Flagged by {' and '.join(reasons)}"
    else:
        escalation_reason = None

    return {
        "url": url,
        "is_confirmed_malicious": is_confirmed_malicious,
        "virustotal_result": vt_res,
        "safe_browsing_result": sb_res,
        "escalation_reason": escalation_reason,
    }


def enrich_detection_with_threat_intel(
    detection_result: Dict[str, Any],
    url: str,
    enable_vt_rate_limit: bool = True,
) -> Dict[str, Any]:
    """
    Enrich an existing ML detection result with threat intelligence cross-validation
    and apply the Section VIII-D escalation rule.

    Escalation Rule:
    ----------------
    Any URL flagged as malicious by VirusTotal or Google Safe Browsing is immediately
    escalated to confirmed malicious/phishing status with 100% confidence, regardless
    of what the ML model originally scored.

    Parameters
    ----------
    detection_result : dict
        Existing detection output dictionary from `phishing_module` or `spam_module`.
    url : str
        The primary URL to cross-validate.
    enable_vt_rate_limit : bool, default=True
        Whether to enforce rate-limiting on VirusTotal queries.

    Returns
    -------
    dict
        Enriched detection result with `threat_intel` and `verdict_source` fields.
    """
    cross_val = cross_validate_url(url, enable_vt_rate_limit=enable_vt_rate_limit)

    enriched = dict(detection_result)
    enriched["threat_intel"] = cross_val

    if cross_val["is_confirmed_malicious"]:
        # Overriding ML verdict due to confirmed threat intelligence
        current_verdict = enriched.get("verdict", "safe").lower()
        if current_verdict not in {"phishing", "malware"}:
            enriched["verdict"] = "phishing"

        enriched["verdict_source"] = "threat_intel_escalation"
        enriched["confidence"] = 1.0
        if "phishing_probability" in enriched:
            enriched["phishing_probability"] = 1.0

        # Inject escalation reason into top features for explainability
        top_feats = list(enriched.get("top_features", []))
        reason = cross_val.get("escalation_reason")
        if reason and reason not in top_feats:
            top_feats.insert(0, reason)
        enriched["top_features"] = top_feats
    else:
        enriched["verdict_source"] = "ml_model"

    return enriched
