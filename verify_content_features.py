"""
Verification Script for NLP Page Content Analysis Extractor
===========================================================
Runs extract_content_features on:
1. Clear phishing passage (urgent tone, immediate password demand, 24h deadline)
2. Neutral legitimate passage (About Us company overview)
3. Borderline legitimate passage (Password reset notice mentioning 'password'/'verify' without pressure)
4. Edge cases (empty string, whitespace, very short text)

Verifies score rankings and assertions.
"""

import sys
import pandas as pd
from phishing_module.content_features import (
    extract_content_features,
    extract_content_features_batch,
)

SAMPLE_1_PHISHING = (
    "Critical security alert: Your account has been restricted due to unauthorized login attempts. "
    "You must act now and immediately verify your account by confirming your password and login credentials. "
    "If you fail to respond within 24 hours, your access will be permanently suspended."
)

SAMPLE_2_NEUTRAL = (
    "Acme Corporation is a leading provider of sustainable architectural materials and thermal insulation products. "
    "Founded in 2012, our multidisciplinary team of engineers collaborates with commercial builders worldwide "
    "to optimize energy efficiency and lower environmental impact."
)

SAMPLE_3_BORDERLINE = (
    "We received a request to reset the password for your account. "
    "If you initiated this change, please click the link below to choose a new password. "
    "If you did not request this, you can safely ignore this email and your existing login settings will remain active."
)


def run_verification():
    print("=" * 80)
    print("RUNNING NLP PAGE CONTENT ANALYSIS VERIFICATION")
    print("=" * 80)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    # 1. Evaluate Individual Samples
    samples = [
        ("Sample 1 (Clear Phishing)", SAMPLE_1_PHISHING),
        ("Sample 2 (Neutral Legitimate)", SAMPLE_2_NEUTRAL),
        ("Sample 3 (Borderline Password Reset)", SAMPLE_3_BORDERLINE),
    ]

    results = []
    print("\n--- Detailed Per-Sample Content Analysis ---")
    for label, text in samples:
        res = extract_content_features(text)
        print(f"\n[{label}]")
        print(f"  Text: \"{text}\"")
        print(f"  • Urgency Score:               {res['urgency_score']:.4f}")
        print(f"  • Credential Request Score:    {res['credential_request_score']:.4f}")
        print(f"  • Login Prompt Density:        {res['login_prompt_density']:.4f}")
        print(f"  • Overall Content Risk Score:  {res['overall_content_risk_score']:.4f}")
        print(f"  • Top TF-IDF Terms:            {res['tfidf_top_terms']}")
        results.append((label, res))

    res1 = results[0][1]
    res2 = results[1][1]
    res3 = results[2][1]

    # 2. Batch Extraction DataFrame
    print("\n" + "=" * 80)
    print("BATCH EXTRACTION DATAFRAME")
    print("=" * 80)
    df_batch = extract_content_features_batch([s[1] for s in samples])
    print(df_batch[["text_preview", "urgency_score", "credential_request_score", "login_prompt_density", "overall_content_risk_score"]].to_string(index=False))

    # 3. Assertions and Comparisons
    print("\n" + "=" * 80)
    print("VERIFYING SCORE RANKINGS AND THRESHOLDS")
    print("=" * 80)

    # Sample 1 must have high urgency, credential request, and overall risk scores
    assert res1["urgency_score"] > 0.4, f"Expected high urgency for Sample 1, got {res1['urgency_score']}"
    assert res1["credential_request_score"] > 0.4, f"Expected high credential request for Sample 1, got {res1['credential_request_score']}"
    assert res1["overall_content_risk_score"] > 0.45, f"Expected high overall risk for Sample 1, got {res1['overall_content_risk_score']}"
    print(f"PASS: Sample 1 (Phishing) correctly scored high overall risk ({res1['overall_content_risk_score']:.4f}).")

    # Sample 2 must have 0 or near-0 risk scores
    assert res2["urgency_score"] == 0.0, f"Expected 0 urgency for Sample 2, got {res2['urgency_score']}"
    assert res2["credential_request_score"] == 0.0, f"Expected 0 credential request for Sample 2, got {res2['credential_request_score']}"
    assert res2["overall_content_risk_score"] == 0.0, f"Expected 0 risk for Sample 2, got {res2['overall_content_risk_score']}"
    print(f"PASS: Sample 2 (Neutral) scored 0.0000 across all risk dimensions.")

    # Sample 3 (Borderline password reset) should have low urgency (0.0) and low overall risk (< 0.25)
    assert res3["urgency_score"] == 0.0, f"Expected 0 urgency for Sample 3, got {res3['urgency_score']}"
    assert res3["overall_content_risk_score"] < 0.25, f"Expected low overall risk for Sample 3, got {res3['overall_content_risk_score']}"
    assert res1["overall_content_risk_score"] > res3["overall_content_risk_score"] * 2.0, (
        f"Sample 1 risk ({res1['overall_content_risk_score']}) should be substantially higher than Sample 3 ({res3['overall_content_risk_score']})"
    )
    print(f"PASS: Sample 3 (Borderline Password Reset) maintained safe low risk ({res3['overall_content_risk_score']:.4f}) without false-positive trigger.")

    # 4. Edge Cases & Robustness
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES & ROBUSTNESS")
    print("=" * 80)

    edge_cases = [
        ("", "Empty String"),
        ("   \n\t  ", "Whitespace only"),
        ("Hello", "Single word"),
        ("Quick test", "Two words"),
        (None, "Non-string None"),
    ]

    for raw_input, desc in edge_cases:
        res_edge = extract_content_features(raw_input)
        assert res_edge["urgency_score"] == 0.0
        assert res_edge["credential_request_score"] == 0.0
        assert res_edge["login_prompt_density"] == 0.0
        assert res_edge["overall_content_risk_score"] == 0.0
        print(f"PASS: Correctly handled edge case [{desc}] -> zeroed-out scores without error.")

    print("\n" + "=" * 80)
    print("ALL NLP CONTENT EXTRACTION TESTS & SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
