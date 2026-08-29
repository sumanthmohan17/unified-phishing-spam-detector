"""
Verification Script for Local FastAPI Backend Server
====================================================
Tests GET /health and POST /analyze-url endpoints across phishing, legitimate,
and fast ML-only scan requests over HTTP.
"""

import json
import pprint
import sys
import time
import requests

# Ensure safe UTF-8 printing on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"


def wait_for_server(timeout: int = 20) -> bool:
    """Wait until the local FastAPI server is healthy and responding."""
    start_time = time.time()
    print(f"Waiting for API server to become ready at {BASE_URL}...")
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                print(f"Server is ready! (Health Check: {resp.json()})")
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_verification():
    print("=" * 80)
    print("RUNNING LOCAL FASTAPI BACKEND SERVER VERIFICATION")
    print("=" * 80)

    if not wait_for_server(timeout=25):
        print("ERROR: API Server did not respond within timeout.", file=sys.stderr)
        sys.exit(1)

    # 1. Health Check Endpoint
    print("\n" + "=" * 80)
    print("1. TESTING GET /health ENDPOINT")
    print("=" * 80)
    resp_health = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Status Code: {resp_health.status_code}")
    print("JSON Response:")
    pprint.pprint(resp_health.json())
    assert resp_health.status_code == 200
    assert resp_health.json().get("status") == "ok"
    assert resp_health.json().get("model_loaded") is True
    print("PASS: GET /health returned status: ok.")

    # 2. Analyze Phishing URL (Full Threat Intel + LLM Explanation)
    print("\n" + "=" * 80)
    print("2. TESTING POST /analyze-url WITH PHISHING URL (Threat Intel + LLM)")
    print("=" * 80)
    phish_payload = {
        "url": "http://192.168.1.100/paypal-security-update/login.php?redirect=bit.ly/secure-auth",
        "check_threat_intel": True,
        "generate_explanation": True,
    }
    print(f"Request Payload:\n{json.dumps(phish_payload, indent=2)}")

    t0 = time.time()
    resp_phish = requests.post(f"{BASE_URL}/analyze-url", json=phish_payload, timeout=20)
    elapsed_phish = time.time() - t0

    print(f"\nResponse Time: {elapsed_phish:.2f}s | Status Code: {resp_phish.status_code}")
    data_phish = resp_phish.json()
    print("Full JSON Response:")
    pprint.pprint(data_phish)

    assert resp_phish.status_code == 200
    assert data_phish["verdict"] == "phishing"
    assert data_phish["confidence"] >= 0.70
    assert len(data_phish["top_features"]) > 0
    assert data_phish["threat_intel"] is not None
    assert data_phish["llm_explanation"] is not None
    assert data_phish["llm_explanation"]["recommended_action"] == "block"
    print("PASS: Phishing URL correctly classified with threat intel and LLM block recommendation.")

    # 3. Analyze Legitimate URL (Full Threat Intel + LLM Explanation)
    print("\n" + "=" * 80)
    print("3. TESTING POST /analyze-url WITH LEGITIMATE URL (Threat Intel + LLM)")
    print("=" * 80)
    legit_payload = {
        "url": "https://www.google.com",
        "check_threat_intel": True,
        "generate_explanation": True,
    }
    print(f"Request Payload:\n{json.dumps(legit_payload, indent=2)}")

    t0 = time.time()
    resp_legit = requests.post(f"{BASE_URL}/analyze-url", json=legit_payload, timeout=20)
    elapsed_legit = time.time() - t0

    print(f"\nResponse Time: {elapsed_legit:.2f}s | Status Code: {resp_legit.status_code}")
    data_legit = resp_legit.json()
    print("Full JSON Response:")
    pprint.pprint(data_legit)

    assert resp_legit.status_code == 200
    assert data_legit["verdict"] == "safe"
    assert data_legit["confidence"] >= 0.70
    assert data_legit["threat_intel"] is not None
    assert data_legit["llm_explanation"] is not None
    assert data_legit["llm_explanation"]["recommended_action"] == "safe"
    print("PASS: Legitimate URL correctly classified with threat intel and LLM safe recommendation.")

    # 4. Analyze URL Fast Mode (check_threat_intel=False, generate_explanation=False)
    print("\n" + "=" * 80)
    print("4. TESTING POST /analyze-url FAST ML-ONLY MODE (Sub-50ms)")
    print("=" * 80)
    fast_payload = {
        "url": "http://192.168.1.55/paypal-security-update/login.php?redirect=bit.ly/auth-fix",
        "check_threat_intel": False,
        "generate_explanation": False,
    }
    print(f"Request Payload:\n{json.dumps(fast_payload, indent=2)}")

    t0 = time.time()
    resp_fast = requests.post(f"{BASE_URL}/analyze-url", json=fast_payload, timeout=5)
    elapsed_fast = time.time() - t0

    print(f"\nResponse Time: {elapsed_fast * 1000:.1f}ms | Status Code: {resp_fast.status_code}")
    data_fast = resp_fast.json()
    print("Full JSON Response:")
    pprint.pprint(data_fast)

    assert resp_fast.status_code == 200
    assert data_fast["verdict"] == "phishing"
    assert data_fast["threat_intel"] is None
    assert data_fast["llm_explanation"] is None
    assert elapsed_fast < 0.20, f"Expected fast sub-200ms ML evaluation, took {elapsed_fast:.3f}s"
    print(f"PASS: Fast ML-only mode responded in {elapsed_fast * 1000:.1f}ms with skipped network calls.")

    print("\n" + "=" * 80)
    print("ALL API SERVER ENDPOINT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
