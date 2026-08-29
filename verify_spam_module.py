"""
Verification Script for Module 2: Email Spam Classification Module
==================================================================
Trains the 4-class Stacking Ensemble on synthetic data, prints metrics,
and tests classify_email on 2 unseen examples (phishing-linked email vs clean legitimate email).
Verifies that embedded URLs are extracted and live-routed into Module 1.
"""

import pprint
import sys
import pandas as pd

from spam_module.classifier import (
    classify_email,
    generate_synthetic_spam_data,
    train_spam_ensemble,
)
from spam_module.preprocessing import preprocess_email_text
from spam_module.url_extraction import extract_urls_from_email


def run_verification():
    print("=" * 80)
    print("RUNNING MODULE 2: EMAIL SPAM CLASSIFICATION & URL ROUTING VERIFICATION")
    print("=" * 80)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    # 1. Preprocessing & URL Extraction Unit Checks
    print("\n--- 1. Testing Preprocessing & URL Extraction ---")
    raw_html_sample = (
        "<html><body><p>Dear customer, please <a href='http://192.168.1.50/secure-login.php'>click here</a> "
        "or visit http://bit.ly/update-now to confirm your details.</p></body></html>"
    )
    cleaned_txt = preprocess_email_text(raw_html_sample)
    extracted_urls = extract_urls_from_email(raw_html_sample)
    print(f"Cleaned Text: \"{cleaned_txt}\"")
    print(f"Extracted URLs: {extracted_urls}")
    assert len(extracted_urls) == 2
    assert "http://192.168.1.50/secure-login.php" in extracted_urls
    assert "http://bit.ly/update-now" in extracted_urls
    print("PASS: Preprocessing and URL extraction work as expected.")

    # 2. Train Stacking Ensemble on Synthetic Data
    print("\n" + "=" * 80)
    print("2. TRAINING 4-CLASS SPAM ENSEMBLE (Naive Bayes + XGBoost -> Meta RF)")
    print("=" * 80)
    df_syn, y_syn = generate_synthetic_spam_data(n_samples=100, random_state=42)
    print(f"Generated {len(df_syn)} synthetic email training samples across 4 classes.")
    print(f"Class distribution: {y_syn.value_counts().to_dict()} (0=Legit, 1=Spam, 2=Phishing, 3=Malware)")

    model, metrics = train_spam_ensemble(df_syn, y_syn, test_size=0.2, random_state=42)
    print("\n--- Model Evaluation Metrics (Held-out 20% Split) ---")
    print(f"  • Overall Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  • Macro Precision:     {metrics['precision_macro']:.4f}")
    print(f"  • Macro Recall:        {metrics['recall_macro']:.4f}")
    print(f"  • Macro F1-Score:      {metrics['f1_macro']:.4f}")

    print("\n--- Per-Class Performance ---")
    for cls_name, cls_metrics in metrics["per_class_metrics"].items():
        print(f"  [{cls_name.upper():<10}] Precision: {cls_metrics['precision']:.4f} | Recall: {cls_metrics['recall']:.4f} | F1: {cls_metrics['f1_score']:.4f}")

    assert metrics["accuracy"] >= 0.85, f"Expected accuracy >= 0.85, got {metrics['accuracy']}"
    print("PASS: 4-class spam stacking ensemble trained successfully.")

    # 3. Test Case 1: Spam Email with Embedded Phishing URL
    print("\n" + "=" * 80)
    print("3. TEST CASE 1: EMAIL WITH EMBEDDED PHISHING URL (Module 1 Live Routing)")
    print("=" * 80)
    phish_email = (
        "CRITICAL SECURITY ALERT: Unauthorized login detected from an unrecognized IP in Moscow. "
        "Your account access has been restricted. Please confirm your credentials immediately to avoid "
        "permanent termination: http://192.168.1.200/paypal-security/login.php?redirect=bit.ly/auth-fix "
        "Failure to act within 24 hours will result in total loss of account access."
    )

    res1 = classify_email(phish_email, model=model)

    print(f"Verdict:        {res1['verdict'].upper()}")
    print(f"Confidence:     {res1['confidence']:.4f}")
    print(f"Embedded URLs:  {res1['embedded_urls']}")
    print(f"Top Features:   {res1['top_features']}")
    print("\n--- Raw embedded_url_risk Dictionary (Live Module 1 Output) ---")
    pprint.pprint(res1["embedded_url_risk"])

    # Assertions for Test Case 1
    assert res1["verdict"] == "phishing", f"Expected verdict 'phishing', got {res1['verdict']}"
    assert len(res1["embedded_urls"]) > 0, "Expected non-empty embedded_urls"
    target_url = res1["embedded_urls"][0]
    assert target_url in res1["embedded_url_risk"], "Expected URL in embedded_url_risk dict"
    url_signals = res1["embedded_url_risk"][target_url]

    # Verify real Module 1 feature extraction outputs
    assert url_signals["has_ip_address"] is True, "Expected has_ip_address = True from Module 1"
    assert url_signals["has_redirect_chain"] is True, "Expected has_redirect_chain = True from Module 1"
    assert "url_entropy" in url_signals, "Expected url_entropy in Module 1 signals"
    print("\nPASS: Embedded URL was successfully extracted and live-routed into Module 1!")

    # 4. Test Case 2: Clean Legitimate Email
    print("\n" + "=" * 80)
    print("4. TEST CASE 2: CLEAN LEGITIMATE EMAIL (No URLs, Neutral Content)")
    print("=" * 80)
    legit_email = (
        "Good afternoon Team,\n\n"
        "Here are the meeting minutes and action items from today's design review. "
        "Please let me know by tomorrow if you have any questions or feedback before we finalize the project timeline.\n\n"
        "Best regards,\nAlex"
    )

    res2 = classify_email(legit_email, model=model)

    print(f"Verdict:        {res2['verdict'].upper()}")
    print(f"Confidence:     {res2['confidence']:.4f}")
    print(f"Embedded URLs:  {res2['embedded_urls']}")
    print(f"Top Features:   {res2['top_features']}")

    # Assertions for Test Case 2
    assert res2["verdict"] == "legitimate", f"Expected verdict 'legitimate', got {res2['verdict']}"
    assert len(res2["embedded_urls"]) == 0, f"Expected empty embedded_urls, got {res2['embedded_urls']}"
    assert res2["confidence"] >= 0.50, f"Expected high confidence for legitimate email, got {res2['confidence']}"
    print("PASS: Clean legitimate email correctly classified with empty embedded_urls.")

    print("\n" + "=" * 80)
    print("ALL MODULE 2 VERIFICATION TESTS & ROUTING ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
