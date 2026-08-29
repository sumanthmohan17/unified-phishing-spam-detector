"""
Verification Script for Threat Intelligence Cross-Validation Module (Module 4)
==============================================================================
Verifies VirusTotal API v3, Google Safe Browsing API v4, cross-validation,
ML verdict escalation override, and sliding-window rate limiting.
"""

import os
import pprint
import sys
import time
from dotenv import load_dotenv

# Ensure safe UTF-8 printing on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from threat_intel.virustotal_check import _vt_limiter, check_url_virustotal
from threat_intel.safe_browsing_check import check_url_safe_browsing
from threat_intel.cross_validator import cross_validate_url, enrich_detection_with_threat_intel


def run_verification():
    print("=" * 80)
    print("RUNNING THREAT INTELLIGENCE CROSS-VALIDATION VERIFICATION (Module 4)")
    print("=" * 80)

    # 1. Check API Keys
    print("\n--- 1. Checking Threat Intelligence API Keys ---")
    vt_key = os.environ.get("VIRUSTOTAL_API_KEY")
    sb_key = os.environ.get("SAFE_BROWSING_API_KEY")

    if vt_key and vt_key.strip() and "your_virustotal_api_key" not in vt_key:
        masked_vt = vt_key[:4] + "..." + vt_key[-4:] if len(vt_key) > 8 else "***"
        print(f"  • VIRUSTOTAL_API_KEY found: {masked_vt} (length: {len(vt_key)})")
    else:
        print("  • VIRUSTOTAL_API_KEY is not configured or contains placeholder.")

    if sb_key and sb_key.strip() and "your_safe_browsing_api_key" not in sb_key:
        masked_sb = sb_key[:4] + "..." + sb_key[-4:] if len(sb_key) > 8 else "***"
        print(f"  • SAFE_BROWSING_API_KEY found: {masked_sb} (length: {len(sb_key)})")
    else:
        print("  • SAFE_BROWSING_API_KEY is not configured or contains placeholder.")

    # 2. Known-Safe URL Check (https://www.google.com)
    print("\n" + "=" * 80)
    print("2. TESTING KNOWN-SAFE URL (https://www.google.com)")
    print("=" * 80)
    safe_url = "https://www.google.com"

    print("\n[Querying VirusTotal API v3...]")
    vt_safe = check_url_virustotal(safe_url, enable_rate_limit=False)
    print(f"  Flagged:          {vt_safe['flagged']}")
    print(f"  Malicious Count:  {vt_safe['malicious_count']} / {vt_safe['total_engines']}")
    if vt_safe["error"]:
        print(f"  Note/Error:       {vt_safe['error']}")

    print("\n[Querying Google Safe Browsing API v4...]")
    sb_safe = check_url_safe_browsing(safe_url)
    print(f"  Flagged:          {sb_safe['flagged']}")
    print(f"  Threat Types:     {sb_safe['threat_types']}")
    if sb_safe["error"]:
        print(f"  Note/Error:       {sb_safe['error']}")

    assert vt_safe["flagged"] is False, "Expected VirusTotal flagged=False for google.com"
    assert sb_safe["flagged"] is False, "Expected Safe Browsing flagged=False for google.com"
    print("\nPASS: Known-safe URL cleared by both VirusTotal and Google Safe Browsing.")

    # 3. Google Safe Browsing Official Malware Test URL
    print("\n" + "=" * 80)
    print("3. TESTING GOOGLE SAFE BROWSING MALWARE TEST URL")
    print("=" * 80)
    malware_test_url = "http://malware.testing.google.test/testing/malware/"
    print(f"URL: {malware_test_url}")

    print("\n[Querying Google Safe Browsing API v4...]")
    sb_mal = check_url_safe_browsing(malware_test_url)
    print(f"  Flagged:          {sb_mal['flagged']}")
    print(f"  Threat Types:     {sb_mal['threat_types']}")
    if sb_mal["error"]:
        print(f"  Note/Error:       {sb_mal['error']}")

    print("\n[Querying VirusTotal API v3...]")
    vt_mal = check_url_virustotal(malware_test_url, enable_rate_limit=False)
    print(f"  Flagged:          {vt_mal['flagged']}")
    print(f"  Malicious Count:  {vt_mal['malicious_count']} / {vt_mal['total_engines']}")
    if vt_mal["error"]:
        print(f"  Note/Error:       {vt_mal['error']}")

    # Assert that Google Safe Browsing caught the official test URL
    assert sb_mal["flagged"] is True, "Expected Safe Browsing to flag official malware test URL"
    assert "MALWARE" in sb_mal["threat_types"], "Expected MALWARE threat type in Safe Browsing output"
    print("\nPASS: Google Safe Browsing correctly flagged official malware test URL.")

    # 4. Test Cross-Validation & Escalation Override
    print("\n" + "=" * 80)
    print("4. TESTING THREAT INTEL ESCALATION RULE OVERRIDING ML VERDICT")
    print("=" * 80)

    # Simulated ML result where ML originally believed the site was safe
    simulated_ml_result = {
        "verdict": "safe",
        "confidence": 0.94,
        "phishing_probability": 0.06,
        "top_features": ["domain_age_days", "uses_https"],
        "raw_scores": {"url_features": {"url_length": 45}},
    }

    print("Original ML Model Result (Pre-Threat Intel):")
    print(f"  • Verdict:        {simulated_ml_result['verdict'].upper()}")
    print(f"  • Confidence:     {simulated_ml_result['confidence']:.2%}")

    enriched = enrich_detection_with_threat_intel(
        simulated_ml_result,
        url=malware_test_url,
        enable_vt_rate_limit=False,
    )

    print("\nEnriched Detection Result (Post-Threat Intel Cross-Validation):")
    print(f"  • Escalated Verdict:     {enriched['verdict'].upper()}")
    print(f"  • Verdict Source:        {enriched['verdict_source']}")
    print(f"  • Confidence:            {enriched['confidence']:.2%}")
    print(f"  • Escalation Reason:     {enriched['threat_intel']['escalation_reason']}")
    print(f"  • Top Features:          {enriched['top_features']}")

    assert enriched["verdict"] == "phishing", "Expected escalated verdict 'phishing'"
    assert enriched["verdict_source"] == "threat_intel_escalation", "Expected verdict_source = 'threat_intel_escalation'"
    assert enriched["confidence"] == 1.0, "Expected confidence = 1.0 upon threat intel escalation"
    assert enriched["threat_intel"]["is_confirmed_malicious"] is True
    print("\nPASS: Escalation rule successfully overridden ML verdict on confirmed threat intel detection!")

    # 5. Test Real Rate Limiter Mechanism on Production _vt_limiter
    print("\n" + "=" * 80)
    print("5. DEMONSTRATING REAL VIRUSTOTAL RATE-LIMITING THROTTLE")
    print("=" * 80)
    print("Exercising production _vt_limiter (4 requests per 60.0s window)...")
    print("Note: Requests 1-4 execute immediately (T+0.00s).")
    print("      Request 5 will wait ~60s to enforce the real 4 req/min free-tier quota window.")
    print("Executing 5 calls on real _vt_limiter...")

    # Clear timestamps for clean test run
    with _vt_limiter._lock:
        _vt_limiter.timestamps.clear()

    start_time = time.time()
    call_times = []

    for i in range(5):
        t_req_start = time.time()
        slept = _vt_limiter.acquire()
        elapsed = time.time() - start_time
        call_times.append(elapsed)
        print(f"  • Request {i + 1}: Executed at T+{elapsed:.2f}s (Slept: {slept:.2f}s)")

    # Confirm the 5th request waited ~60s
    assert call_times[4] >= 59.0, f"Expected 5th request to wait ~60s for sliding window, got {call_times[4]:.2f}s"
    print("\nPASS: Production _vt_limiter correctly enforced real 4 requests/60s rate limit.")

    print("\n" + "=" * 80)
    print("ALL THREAT INTELLIGENCE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
