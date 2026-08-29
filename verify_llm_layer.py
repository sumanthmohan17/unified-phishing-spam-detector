"""
Verification Script for LLM Explainability Layer (Module 3)
===========================================================
Tests explain_phishing_detection and explain_spam_detection using Groq API
(or verifies graceful error handling & fallback when API key is unconfigured).
"""

import os
import pprint
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure safe UTF-8 printing on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import phishing_module
from phishing_module.classifier import generate_synthetic_training_data, train_ensemble, predict
from phishing_module.visual_similarity import build_reference_library

import spam_module
from spam_module.classifier import generate_synthetic_spam_data, train_spam_ensemble, classify_email

from llm_layer.explainer import explain_phishing_detection, explain_spam_detection

BASE_DIR = Path(__file__).parent
REFERENCE_DIR = BASE_DIR / "phishing_module" / "test_assets" / "reference_logos"
QUERY_DIR = BASE_DIR / "phishing_module" / "test_assets" / "test_queries"


def run_verification():
    print("=" * 80)
    print("RUNNING LLM EXPLAINABILITY LAYER VERIFICATION (Module 3)")
    print("=" * 80)

    # 1. Check API Key Status
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key and api_key.strip() and "your_groq_api_key" not in api_key:
        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"\n[KEY STATUS] GROQ_API_KEY found in environment/dotenv: {masked} (length: {len(api_key)})")
        api_ready = True
    else:
        print("\n[KEY STATUS] GROQ_API_KEY is not configured in .env or environment.")
        print("Note: LLM calls will demonstrate graceful error handling and deterministic fallback.")
        api_ready = False

    # 2. Setup Models
    print("\n--- Initializing Module 1 & Module 2 Classifiers ---")
    X_phish, y_phish = generate_synthetic_training_data(100, random_state=42)
    phish_model, _ = train_ensemble(X_phish, y_phish, random_state=42)
    ref_lib = build_reference_library(str(REFERENCE_DIR))

    df_spam, y_spam = generate_synthetic_spam_data(100, random_state=42)
    spam_model, _ = train_spam_ensemble(df_spam, y_spam, random_state=42)
    print("Models initialized successfully.")

    # 3. Test Phishing Explanation (Module 1 Phishing Test Case)
    print("\n" + "=" * 80)
    print("1. EXPLAINING MODULE 1 PHISHING DETECTION")
    print("=" * 80)
    phish_url = "http://192.168.1.100/paypal-security-update/login.php?redirect=bit.ly/secure-auth"
    phish_screenshot = str(QUERY_DIR / "clone_screenshot.png")
    phish_text = (
        "Urgent security alert: Your PayPal account has been restricted due to unauthorized login attempts. "
        "You must act now and immediately verify your account by confirming your password and login credentials. "
        "If you do not respond within 24 hours, your account will be permanently suspended."
    )

    pred_phish = predict(
        url=phish_url,
        screenshot_path=phish_screenshot,
        page_text=phish_text,
        model=phish_model,
        reference_dir=str(REFERENCE_DIR),
        precomputed_library=ref_lib,
    )

    print(f"Verdict:              {pred_phish['verdict'].upper()}")
    print(f"Confidence:           {pred_phish['confidence']:.2%}")
    print(f"Top Features:         {pred_phish['top_features']}")

    expl_phish = explain_phishing_detection(pred_phish)
    print(f"\n[LLM Explanation]:")
    print(f"  \"{expl_phish['explanation']}\"")
    print(f"\n[Recommended Action]: {expl_phish['recommended_action'].upper()}")
    assert expl_phish["recommended_action"] in {"block", "review", "safe"}
    assert len(expl_phish["explanation"]) > 0

    # 4. Test Legitimate Explanation (Module 1 Safe Test Case)
    print("\n" + "=" * 80)
    print("2. EXPLAINING MODULE 1 LEGITIMATE DETECTION")
    print("=" * 80)
    legit_url = "https://www.google.com/about/products"
    legit_screenshot = str(REFERENCE_DIR / "google_logo.png")
    legit_text = "Welcome to Google! Explore our products and services designed to help you search information."

    pred_legit = predict(
        url=legit_url,
        screenshot_path=legit_screenshot,
        page_text=legit_text,
        model=phish_model,
        reference_dir=str(REFERENCE_DIR),
        precomputed_library=ref_lib,
    )

    print(f"Verdict:              {pred_legit['verdict'].upper()}")
    print(f"Confidence:           {pred_legit['confidence']:.2%}")
    print(f"Top Features:         {pred_legit['top_features']}")

    expl_legit = explain_phishing_detection(pred_legit)
    print(f"\n[LLM Explanation]:")
    print(f"  \"{expl_legit['explanation']}\"")
    print(f"\n[Recommended Action]: {expl_legit['recommended_action'].upper()}")
    assert expl_legit["recommended_action"] in {"block", "review", "safe"}
    assert len(expl_legit["explanation"]) > 0

    # 5. Test Spam / Phishing Email Explanation (Module 2 Detection)
    print("\n" + "=" * 80)
    print("3. EXPLAINING MODULE 2 EMAIL SPAM & EMBEDDED URL THREAT DETECTION")
    print("=" * 80)
    spam_email = (
        "CRITICAL SECURITY ALERT: Unauthorized login detected from an unrecognized IP in Moscow. "
        "Your account access has been restricted. Please confirm your credentials immediately to avoid "
        "permanent termination: http://192.168.1.200/paypal-security/login.php?redirect=bit.ly/auth-fix "
        "Failure to act within 24 hours will result in total loss of account access."
    )

    pred_spam = classify_email(spam_email, model=spam_model)
    print(f"Verdict:              {pred_spam['verdict'].upper()}")
    print(f"Confidence:           {pred_spam['confidence']:.2%}")
    print(f"Embedded URLs:        {pred_spam['embedded_urls']}")
    print(f"Top Features:         {pred_spam['top_features']}")

    expl_spam = explain_spam_detection(pred_spam)
    print(f"\n[LLM Explanation]:")
    print(f"  \"{expl_spam['explanation']}\"")
    print(f"\n[Recommended Action]: {expl_spam['recommended_action'].upper()}")
    assert expl_spam["recommended_action"] in {"block", "review", "safe"}
    assert len(expl_spam["explanation"]) > 0

    print("\n" + "=" * 80)
    print("ALL LLM EXPLAINABILITY LAYER VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
