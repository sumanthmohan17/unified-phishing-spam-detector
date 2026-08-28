"""
Verification Script for Multi-Signal Phishing Ensemble Classifier
================================================================
Trains the Stacking Ensemble on synthetic data, prints evaluation metrics,
and runs predict() on 2 unseen end-to-end examples (phishing vs legitimate).
"""

import os
import sys
from pathlib import Path
import pandas as pd

from phishing_module.classifier import (
    generate_synthetic_training_data,
    predict,
    train_ensemble,
    build_feature_vector,
)
from phishing_module.visual_similarity import build_reference_library

BASE_DIR = Path(__file__).parent
REFERENCE_DIR = BASE_DIR / "phishing_module" / "test_assets" / "reference_logos"
QUERY_DIR = BASE_DIR / "phishing_module" / "test_assets" / "test_queries"


def run_verification():
    print("=" * 80)
    print("RUNNING MULTI-SIGNAL PHISHING ENSEMBLE CLASSIFIER VERIFICATION")
    print("=" * 80)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    # 1. Generate Synthetic Training Data
    print("\n--- 1. Generating Synthetic Training Dataset (~100 samples) ---")
    X_syn, y_syn = generate_synthetic_training_data(n_samples=100, random_state=42)
    print(f"Generated {len(X_syn)} synthetic samples with {X_syn.shape[1]} fused features.")
    print(f"Label distribution: {y_syn.value_counts().to_dict()} (1=Phishing, 0=Legitimate)")

    # 2. Train Ensemble Model
    print("\n--- 2. Training Stacking Ensemble (XGBoost + Random Forest Base -> Meta RF) ---")
    model, metrics = train_ensemble(X_syn, y_syn, test_size=0.2, random_state=42)
    print("\nTraining / Validation Metrics on Held-out 20% Split:")
    for metric_name, val in metrics.items():
        print(f"  • {metric_name.capitalize():<12}: {val:.4f}")

    assert metrics["accuracy"] >= 0.85, f"Expected accuracy >= 0.85, got {metrics['accuracy']}"
    print("PASS: Stacking ensemble trained successfully with valid performance metrics.")

    # 3. Build/Load Reference Library
    ref_lib = build_reference_library(str(REFERENCE_DIR))
    print(f"\nLoaded {len(ref_lib)} reference brand entries from: {REFERENCE_DIR}")

    # 4. Predict on Unseen Phishing Example
    print("\n" + "=" * 80)
    print("3. END-TO-END PREDICTION ON UNSEEN PHISHING TEST CASE")
    print("=" * 80)

    phish_url = "http://192.168.1.100/paypal-security-update/login.php?redirect=bit.ly/secure-auth"
    phish_screenshot = str(QUERY_DIR / "clone_screenshot.png")
    phish_text = (
        "Urgent security alert: Your PayPal account has been restricted due to unauthorized login attempts. "
        "You must act now and immediately verify your account by confirming your password and login credentials. "
        "If you do not respond within 24 hours, your account will be permanently suspended."
    )

    res_phish = predict(
        url=phish_url,
        screenshot_path=phish_screenshot,
        page_text=phish_text,
        model=model,
        reference_dir=str(REFERENCE_DIR),
        precomputed_library=ref_lib,
    )

    print(f"URL: {phish_url}")
    print(f"Verdict:              {res_phish['verdict'].upper()}")
    print(f"Confidence:           {res_phish['confidence']:.4f}")
    print(f"Phishing Probability: {res_phish['phishing_probability']:.4f}")
    print(f"Top Features:         {res_phish['top_features']}")
    print(f"Raw Signals Summary:")
    print(f"  • URL:    Length={res_phish['raw_scores']['url_features']['url_length']}, HasIP={res_phish['raw_scores']['url_features']['has_ip_address']}, Redirect={res_phish['raw_scores']['url_features']['has_redirect_chain']}")
    print(f"  • Visual: Match={res_phish['raw_scores']['visual_features']['closest_match']}, Distance={res_phish['raw_scores']['visual_features']['hash_distance']}, IsSimilar={res_phish['raw_scores']['visual_features']['is_visually_similar']}")
    print(f"  • NLP:    Urgency={res_phish['raw_scores']['content_features']['urgency_score']:.4f}, Credential={res_phish['raw_scores']['content_features']['credential_request_score']:.4f}, RiskScore={res_phish['raw_scores']['content_features']['overall_content_risk_score']:.4f}")

    assert res_phish["verdict"] == "phishing", "Expected phishing verdict for Test Case 1"
    assert res_phish["confidence"] >= 0.70, f"Expected high confidence for phishing, got {res_phish['confidence']}"
    assert len(res_phish["top_features"]) > 0, "Expected non-empty top_features"
    assert any(flag in res_phish["top_features"] for flag in ["has_ip_address", "has_redirect_chain", "is_visually_similar"]), (
        f"Expected at least one binary red-flag (has_ip_address, has_redirect_chain, is_visually_similar) in top_features, got {res_phish['top_features']}"
    )
    print("PASS: Unseen Phishing test case correctly classified with high confidence and red-flag explainability.")

    # 5. Predict on Unseen Legitimate Example
    print("\n" + "=" * 80)
    print("4. END-TO-END PREDICTION ON UNSEEN LEGITIMATE TEST CASE")
    print("=" * 80)

    legit_url = "https://www.google.com/about/products"
    legit_screenshot = str(REFERENCE_DIR / "google_logo.png")
    legit_text = (
        "Welcome to Google. Explore our wide range of products, technology innovations, "
        "and developer documentation designed to help organize the world's information and make it universally accessible."
    )

    res_legit = predict(
        url=legit_url,
        screenshot_path=legit_screenshot,
        page_text=legit_text,
        model=model,
        reference_dir=str(REFERENCE_DIR),
        precomputed_library=ref_lib,
    )

    print(f"URL: {legit_url}")
    print(f"Verdict:              {res_legit['verdict'].upper()}")
    print(f"Confidence:           {res_legit['confidence']:.4f}")
    print(f"Phishing Probability: {res_legit['phishing_probability']:.4f}")
    print(f"Top Features:         {res_legit['top_features']}")
    print(f"Raw Signals Summary:")
    print(f"  • URL:    Length={res_legit['raw_scores']['url_features']['url_length']}, HasIP={res_legit['raw_scores']['url_features']['has_ip_address']}, UsesHTTPS={res_legit['raw_scores']['url_features']['uses_https']}")
    print(f"  • Visual: Match={res_legit['raw_scores']['visual_features']['closest_match']}, Distance={res_legit['raw_scores']['visual_features']['hash_distance']}, IsSimilar={res_legit['raw_scores']['visual_features']['is_visually_similar']}")
    print(f"  • NLP:    Urgency={res_legit['raw_scores']['content_features']['urgency_score']:.4f}, Credential={res_legit['raw_scores']['content_features']['credential_request_score']:.4f}, RiskScore={res_legit['raw_scores']['content_features']['overall_content_risk_score']:.4f}")

    assert res_legit["verdict"] == "safe", "Expected safe verdict for Test Case 2"
    assert res_legit["confidence"] >= 0.70, f"Expected high confidence for legitimate site, got {res_legit['confidence']}"
    print("PASS: Unseen Legitimate test case correctly classified with high confidence.")

    print("\n" + "=" * 80)
    print("ALL ENSEMBLE CLASSIFIER VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
