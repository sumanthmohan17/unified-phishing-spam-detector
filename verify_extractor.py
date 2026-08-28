"""
Verification Script for URL Structural Feature Extractor
========================================================
Runs feature extraction on the sample URLs from the prompt, verifies false-positive
edge cases (e.g. protect.com, test.company.com, this.gdrive.com), and asserts
redirect chain logic.
"""

import sys
import pandas as pd
from phishing_module.feature_extraction import (
    extract_url_features,
    extract_url_features_batch,
    validate_url,
)

SAMPLE_URLS = [
    "https://www.google.com",
    "http://192.168.1.1/login/secure-update.php",
    "https://paypa1-verify-account.tk/signin?redirect=bit.ly/xk29Fj",
    "https://accounts.google.com/signin/v2/identifier",
    "http://bit.ly/3xK9mZq",
]

# False-positive test cases: domains that contain shortener substrings ('t.co', 'is.gd') but are legitimate domains
FALSE_POSITIVE_TEST_CASES = [
    "https://protect.com",
    "https://test.company.com",
    "https://this.gdrive.com",
]


def run_verification():
    print("=" * 80)
    print("1. RUNNING ORIGINAL 5 SAMPLE URLS VERIFICATION")
    print("=" * 80)

    # Configure pandas display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", lambda x: "%.4f" % x)

    # 1. Batch extraction on sample URLs
    df = extract_url_features_batch(SAMPLE_URLS)
    print("\n--- Sample URLs DataFrame ---")
    print(df.to_string(index=False))

    # Assert expected has_redirect_chain values for sample URLs
    expected_redirect_samples = {
        "https://www.google.com": False,
        "http://192.168.1.1/login/secure-update.php": False,
        "https://paypa1-verify-account.tk/signin?redirect=bit.ly/xk29Fj": True,
        "https://accounts.google.com/signin/v2/identifier": False,
        "http://bit.ly/3xK9mZq": True,
    }

    for url, expected in expected_redirect_samples.items():
        actual = bool(df.loc[df["url"] == url, "has_redirect_chain"].iloc[0])
        assert actual == expected, f"Assertion failed for {url}: expected {expected}, got {actual}"
        print(f"PASS: {url} -> has_redirect_chain = {actual} (expected: {expected})")

    # 2. Test False Positive Cases
    print("\n" + "=" * 80)
    print("2. TESTING FALSE-POSITIVE DOMAIN CASES (protect.com, test.company.com, this.gdrive.com)")
    print("=" * 80)

    df_fp = extract_url_features_batch(FALSE_POSITIVE_TEST_CASES)
    print("\n--- False Positive Test Cases DataFrame ---")
    print(df_fp.to_string(index=False))

    for url in FALSE_POSITIVE_TEST_CASES:
        actual = bool(df_fp.loc[df_fp["url"] == url, "has_redirect_chain"].iloc[0])
        assert actual is False, f"False positive detected for {url}: has_redirect_chain should be False, but got {actual}"
        print(f"PASS: {url} -> has_redirect_chain = {actual} (correctly False)")

    # 3. Test Input Validation & Edge Cases
    print("\n" + "=" * 80)
    print("3. TESTING INPUT VALIDATION & ERROR HANDLING")
    print("=" * 80)

    invalid_test_cases = [
        ("", "Empty URL"),
        ("   ", "Whitespace only"),
        ("not_a_valid_url", "Missing scheme"),
        (12345, "Non-string input"),
        ("http://", "Missing hostname"),
    ]

    for val, case_name in invalid_test_cases:
        try:
            extract_url_features(val)
            print(f"FAIL: Expected error for case '{case_name}' with value '{val}', but none was raised.")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            print(f"PASS: Correctly caught invalid input [{case_name}]: {e}")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS, FALSE-POSITIVE CHECKS, AND ASSERTIONS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
