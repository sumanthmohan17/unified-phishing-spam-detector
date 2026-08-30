"""
Verification Script for Real-World Phishing Benchmark Dataset Training
======================================================================
Sources raw URLs from UCI PhiUSIIL Phishing URL Dataset (id=967) and
augments with ~100 realistic complex-path legitimate URLs from top domains.
Extracts real structural features with live WHOIS queries, verifies
distribution parity, trains the Stacking Ensemble, and specifically validates
that complex legitimate URLs (github.com/microsoft/vscode, amazon.com/dp/...,
claude.ai/chat/...) correctly receive the "safe" verdict without false positives.
"""

from __future__ import annotations

import sys
import time
import pandas as pd

# Ensure safe UTF-8 printing on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

from phishing_module.real_data_loader import fetch_and_label_urls, build_real_training_dataset
from phishing_module.classifier import train_ensemble, predict


def run_verification():
    print("=" * 80)
    print("REAL-WORLD PHISHING BENCHMARK TRAINING (AUGMENTED COMPLEX-PATH PARITY)")
    print("=" * 80)

    # 1. Fetch raw URLs from UCI PhiUSIIL Dataset + Augmented Complex Legitimate URLs
    print("\n--- 1. Fetching Real Benchmark URLs & Augmented Complex Legitimate Set ---")
    t_fetch_start = time.time()
    urls, labels = fetch_and_label_urls(
        n_phishing=300,
        n_legitimate=300,
        n_complex_legitimate=100,
        random_state=42,
    )
    fetch_time = time.time() - t_fetch_start

    n_phish = sum(1 for y in labels if y == 1)
    n_legit = sum(1 for y in labels if y == 0)

    print(f"Fetch completed in {fetch_time:.2f}s:")
    print(f"  • Verified Phishing URLs (PhiUSIIL):          {n_phish}")
    print(f"  • Standard Legitimate URLs (PhiUSIIL):         300")
    print(f"  • Complex-Path Legitimate URLs (Augmented):    100")
    print(f"  • Total Legitimate URLs:                       {n_legit}")
    print(f"  • Total Combined Labeled URLs:                 {len(urls)}")

    assert n_phish == 300, f"Expected 300 phishing URLs, got {n_phish}"
    assert n_legit == 400, f"Expected 400 legitimate URLs, got {n_legit}"

    # 2. Extract Real Features with live WHOIS queries
    print("\n" + "=" * 80)
    print("2. EXTRACTING REAL STRUCTURAL FEATURES & WHOIS DATA")
    print("=" * 80)
    print(f"Extracting 16-signal feature vectors across {len(urls)} URLs...")
    print("Note: Feature extraction performs live DNS/WHOIS resolution.\n")

    t_feat_start = time.time()
    X, y = build_real_training_dataset(urls=urls, labels=labels, use_cache=True)
    feat_time = time.time() - t_feat_start

    print(f"\nFeature extraction / load completed in {feat_time:.2f}s.")
    print(f"Extracted Dataset Shape: X = {X.shape}, y = {y.shape}")

    # 3. Inspect Sample Extracted Feature Data
    print("\n" + "=" * 80)
    print("3. SAMPLE EXTRACTED FEATURE DATA (5 Rows)")
    print("=" * 80)
    sample_df = X[["url_length", "url_entropy", "domain_age_days", "special_char_count", "has_ip_address", "has_redirect_chain", "uses_https", "hyphen_count"]].copy()
    sample_df["label"] = y
    sample_df["label_desc"] = sample_df["label"].map({1: "PHISHING", 0: "LEGITIMATE"})
    print(sample_df.head(5).to_string())

    # 4. Feature Distribution Breakdown (Phishing vs Legitimate)
    print("\n" + "=" * 80)
    print("4. FEATURE DISTRIBUTION COMPARISON (Phishing vs Legitimate)")
    print("=" * 80)
    print("Confirming feature distributions across both classes to verify structural parity:\n")

    dist_features = [
        "url_length",
        "special_char_count",
        "url_entropy",
        "subdomain_count",
        "domain_age_days",
        "hyphen_count",
        "has_ip_address",
        "uses_https",
    ]

    phish_mask = (y == 1)
    legit_mask = (y == 0)

    rows = []
    for feat in dist_features:
        p_vals = X.loc[phish_mask, feat]
        l_vals = X.loc[legit_mask, feat]
        rows.append({
            "Feature": feat,
            "Phishing Mean (Std)": f"{p_vals.mean():.2f} (±{p_vals.std():.2f})",
            "Phishing Range": f"[{p_vals.min():.0f}, {p_vals.max():.0f}]",
            "Legit Mean (Std)": f"{l_vals.mean():.2f} (±{l_vals.std():.2f})",
            "Legit Range": f"[{l_vals.min():.0f}, {l_vals.max():.0f}]",
        })

    dist_table = pd.DataFrame(rows)
    print(dist_table.to_string(index=False))

    # 5. Train Stacking Ensemble on Augmented Real Data
    print("\n" + "=" * 80)
    print("5. TRAINING STACKING ENSEMBLE CLASSIFIER ON REAL DATA")
    print("=" * 80)
    t_train_start = time.time()
    model, metrics = train_ensemble(X, y, test_size=0.20, random_state=42)
    train_time = time.time() - t_train_start

    print(f"Ensemble training completed in {train_time:.2f}s.\n")
    print("Real Empirical Performance on Held-Out Test Split (20% Validation / 140 URLs):")
    print(f"  • Accuracy:   {metrics['accuracy']:.2%} ({metrics['accuracy']})")
    print(f"  • Precision:  {metrics['precision']:.2%} ({metrics['precision']})")
    print(f"  • Recall:     {metrics['recall']:.2%} ({metrics['recall']})")
    print(f"  • F1-Score:   {metrics['f1_score']:.2%} ({metrics['f1_score']})")

    # 6. Base Estimator Feature Importances
    print("\n" + "=" * 80)
    print("6. LEARNED FEATURE IMPORTANCES (Real Data Distribution)")
    print("=" * 80)
    xgb_est = model.named_estimators_["xgb"]
    rf_est = model.named_estimators_["rf"]

    feat_names = list(X.columns)
    importances_xgb = dict(zip(feat_names, xgb_est.feature_importances_))
    importances_rf = dict(zip(feat_names, rf_est.feature_importances_))

    top_xgb = sorted(importances_xgb.items(), key=lambda x: x[1], reverse=True)[:5]
    top_rf = sorted(importances_rf.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 Features - XGBoost Base Estimator:")
    for feat, imp in top_xgb:
        print(f"  • {feat:<28}: {imp:.4f}")

    print("\nTop 5 Features - Random Forest Base Estimator:")
    for feat, imp in top_rf:
        print(f"  • {feat:<28}: {imp:.4f}")

    # 7. False-Positive Validation on Complex Legitimate URLs
    print("\n" + "=" * 80)
    print("7. FALSE-POSITIVE VALIDATION ON COMPLEX LEGITIMATE URLS")
    print("=" * 80)

    test_cases = [
        {
            "name": "GitHub Repo & File Path",
            "url": "https://github.com/microsoft/vscode",
            "expected_verdict": "safe",
        },
        {
            "name": "Amazon Product Detail Page",
            "url": "https://www.amazon.com/dp/B08N5WRWNW",
            "expected_verdict": "safe",
        },
        {
            "name": "Claude AI Chat UUID Path",
            "url": "https://claude.ai/chat/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "expected_verdict": "safe",
        },
        {
            "name": "FastAPI Technical Documentation with Anchor",
            "url": "https://fastapi.tiangolo.com/tutorial/bigger-applications/#an-example-file-structure",
            "expected_verdict": "safe",
        },
        {
            "name": "Standard Search Engine",
            "url": "https://www.google.com",
            "expected_verdict": "safe",
        },
        {
            "name": "Suspicious Raw IP Phishing Attack",
            "url": "http://192.168.1.100/paypal-security-update/login.php?redirect=bit.ly/secure-auth",
            "expected_verdict": "phishing",
        },
        {
            "name": "Shortener Redirect Phishing URL",
            "url": "http://bit.ly/secure-paypal-login-update-account",
            "expected_verdict": "phishing",
        },
    ]

    all_passed = True
    for tc in test_cases:
        pred = predict(url=tc["url"], model=model)
        status = "PASS" if pred["verdict"] == tc["expected_verdict"] else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"\n[{status}] Test Case: {tc['name']}")
        print(f"  URL:         {tc['url']}")
        print(f"  Expected:    {tc['expected_verdict'].upper()} | Predicted: {pred['verdict'].upper()} (Confidence: {pred['confidence']:.1%}, PhishProb: {pred['phishing_probability']:.1%})")
        print(f"  Top Signals: {pred['top_features']}")

    assert all_passed, "One or more complex URL test cases failed!"

    print("\n" + "=" * 80)
    print("ALL REAL-WORLD BENCHMARK EVALUATIONS & FALSE-POSITIVE TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
