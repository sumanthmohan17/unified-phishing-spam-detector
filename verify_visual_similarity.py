"""
Verification Script for Visual Brand Similarity Extractor
=========================================================
Generates synthetic reference logos and test screenshots with Pillow,
and runs compare_screenshot to verify:
1. Precomputed reference library generation.
2. Cloned screenshot on a deceptive domain ('paypal-secure-login.tk') -> is_visually_similar = True.
3. Same screenshot on the real registered domain ('paypal.com') -> is_visually_similar = False.
4. Same screenshot on a subdomain of the real domain ('www.paypal.com') -> is_visually_similar = False.
5. Reference image matched against itself -> hash_distance = 0.
6. Completely distinct image -> high hash_distance > 10, is_visually_similar = False.
7. Graceful error handling on missing/corrupt image files and non-existent folders.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

from phishing_module.visual_similarity import (
    build_reference_library,
    compare_screenshot,
    extract_implied_brand,
)

BASE_DIR = Path(__file__).parent
REFERENCE_DIR = BASE_DIR / "phishing_module" / "test_assets" / "reference_logos"
QUERY_DIR = BASE_DIR / "phishing_module" / "test_assets" / "test_queries"


def generate_synthetic_assets():
    """Generate synthetic reference brand logos and test query images."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. PayPal Reference Logo
    paypal_img = Image.new("RGB", (200, 200), color=(0, 48, 135))  # Deep Blue
    draw = ImageDraw.Draw(paypal_img)
    draw.rectangle([40, 40, 160, 160], fill=(0, 121, 193), outline=(255, 255, 255), width=3)
    draw.text((65, 90), "PayPal", fill=(255, 255, 255))
    paypal_path = REFERENCE_DIR / "paypal_logo.png"
    paypal_img.save(paypal_path)

    # 2. Google Reference Logo
    google_img = Image.new("RGB", (200, 200), color=(255, 255, 255))  # White
    draw = ImageDraw.Draw(google_img)
    draw.ellipse([30, 30, 90, 90], fill=(234, 67, 53))    # Red
    draw.ellipse([110, 30, 170, 90], fill=(66, 133, 244))  # Blue
    draw.ellipse([30, 110, 90, 170], fill=(251, 188, 5))   # Yellow
    draw.ellipse([110, 110, 170, 170], fill=(52, 168, 83)) # Green
    draw.text((75, 95), "Google", fill=(0, 0, 0))
    google_path = REFERENCE_DIR / "google_logo.png"
    google_img.save(google_path)

    # 3. Microsoft Reference Logo
    msft_img = Image.new("RGB", (200, 200), color=(240, 240, 240))  # Light Grey
    draw = ImageDraw.Draw(msft_img)
    draw.rectangle([40, 40, 95, 95], fill=(242, 80, 34))    # Red
    draw.rectangle([105, 40, 160, 95], fill=(127, 186, 0))  # Green
    draw.rectangle([40, 105, 95, 160], fill=(0, 164, 239))  # Blue
    draw.rectangle([105, 105, 160, 160], fill=(255, 185, 0)) # Yellow
    msft_path = REFERENCE_DIR / "microsoft_logo.png"
    msft_img.save(msft_path)

    # 4. Clone Screenshot (PayPal visual clone)
    clone_img = paypal_img.copy()
    clone_path = QUERY_DIR / "clone_screenshot.png"
    clone_img.save(clone_path)

    # 5. Completely Different Query Image
    diff_img = Image.new("RGB", (200, 200), color=(34, 139, 34))  # Forest Green
    draw = ImageDraw.Draw(diff_img)
    draw.polygon([(100, 20), (180, 180), (20, 180)], fill=(139, 69, 19))  # Brown Triangle
    draw.ellipse([80, 80, 120, 120], fill=(255, 215, 0))  # Gold Sun
    diff_path = QUERY_DIR / "unrelated_forest_landscape.png"
    diff_img.save(diff_path)

    # 6. Corrupt image file
    corrupt_path = QUERY_DIR / "corrupted_image.png"
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPTED_NON_IMAGE_DATA_BYTES")


def run_verification():
    print("=" * 80)
    print("RUNNING VISUAL BRAND SIMILARITY EXTRACTION VERIFICATION")
    print("=" * 80)

    generate_synthetic_assets()

    # Step 1: Precompute reference library
    print("\n" + "=" * 80)
    print("1. PRECOMPUTING REFERENCE LIBRARY CACHE")
    print("=" * 80)
    ref_lib = build_reference_library(str(REFERENCE_DIR))
    print(f"Precomputed {len(ref_lib)} reference brand entries:")
    for fname, meta in ref_lib.items():
        print(f"  • {fname}: Implied Brand='{meta['implied_brand']}', pHash={meta['hash']}")

    # Step 2: Test Clone on Deceptive Domain (paypal-secure-login.tk) -> Expect is_visually_similar = True
    print("\n" + "=" * 80)
    print("2. TEST CLONED SCREENSHOT ON DECEPTIVE DOMAIN (paypal-secure-login.tk)")
    print("=" * 80)
    clone_query = str(QUERY_DIR / "clone_screenshot.png")
    res_phish = compare_screenshot(
        image_path=clone_query,
        reference_dir=str(REFERENCE_DIR),
        threshold=10,
        precomputed_library=ref_lib,
        current_domain="paypal-secure-login.tk",
    )
    print(f"Query Image: {Path(clone_query).name}")
    print(f"Current Domain: paypal-secure-login.tk")
    print(f"Result:")
    print(f"  • Closest Match: {res_phish['closest_match']}")
    print(f"  • Hash Distance: {res_phish['hash_distance']}")
    print(f"  • Is Visually Similar (Phishing Mismatch Flag): {res_phish['is_visually_similar']}")
    print(f"  • All Distances: {res_phish['all_distances']}")

    assert res_phish["closest_match"] == "paypal_logo.png"
    assert res_phish["hash_distance"] == 0
    assert res_phish["is_visually_similar"] is True, "Expected True: cloned visual on deceptive domain must be flagged!"
    print("PASS: Successfully flagged phishing impersonation on 'paypal-secure-login.tk' (is_visually_similar = True).")

    # Step 3: Test Clone on Authentic Domain (paypal.com) -> Expect is_visually_similar = False
    print("\n" + "=" * 80)
    print("3. TEST CLONED SCREENSHOT ON REAL REGISTERED DOMAIN (paypal.com)")
    print("=" * 80)
    res_legit = compare_screenshot(
        image_path=clone_query,
        reference_dir=str(REFERENCE_DIR),
        threshold=10,
        precomputed_library=ref_lib,
        current_domain="paypal.com",
    )
    print(f"Query Image: {Path(clone_query).name}")
    print(f"Current Domain: paypal.com")
    print(f"Result:")
    print(f"  • Closest Match: {res_legit['closest_match']}")
    print(f"  • Hash Distance: {res_legit['hash_distance']}")
    print(f"  • Is Visually Similar (Phishing Mismatch Flag): {res_legit['is_visually_similar']}")

    assert res_legit["closest_match"] == "paypal_logo.png"
    assert res_legit["hash_distance"] == 0
    assert res_legit["is_visually_similar"] is False, "Expected False: authentic domain should not be flagged as phishing."
    print("PASS: Correctly recognized real registered domain 'paypal.com' (is_visually_similar = False).")

    # Step 4: Test Clone on Subdomain of Authentic Domain (www.paypal.com) -> Expect is_visually_similar = False
    print("\n" + "=" * 80)
    print("4. TEST CLONED SCREENSHOT ON SUBDOMAIN OF REAL DOMAIN (www.paypal.com)")
    print("=" * 80)
    res_subdomain = compare_screenshot(
        image_path=clone_query,
        reference_dir=str(REFERENCE_DIR),
        threshold=10,
        precomputed_library=ref_lib,
        current_domain="www.paypal.com",
    )
    print(f"Query Image: {Path(clone_query).name}")
    print(f"Current Domain: www.paypal.com")
    print(f"Result:")
    print(f"  • Closest Match: {res_subdomain['closest_match']}")
    print(f"  • Hash Distance: {res_subdomain['hash_distance']}")
    print(f"  • Is Visually Similar (Phishing Mismatch Flag): {res_subdomain['is_visually_similar']}")

    assert res_subdomain["closest_match"] == "paypal_logo.png"
    assert res_subdomain["hash_distance"] == 0
    assert res_subdomain["is_visually_similar"] is False, "Expected False: subdomain of authentic domain should not be flagged."
    print("PASS: Correctly resolved subdomain 'www.paypal.com' to registered domain 'paypal.com' (is_visually_similar = False).")

    # Step 5: Test on Synthetic Reference Image Itself
    print("\n" + "=" * 80)
    print("5. TEST ON SYNTHETIC REFERENCE IMAGE ITSELF (paypal_logo.png)")
    print("=" * 80)
    ref_img_query = str(REFERENCE_DIR / "paypal_logo.png")
    res_self = compare_screenshot(
        image_path=ref_img_query,
        reference_dir=str(REFERENCE_DIR),
        threshold=10,
        precomputed_library=ref_lib,
    )
    print(f"Query Image: {Path(ref_img_query).name}")
    print(f"Result:")
    print(f"  • Closest Match: {res_self['closest_match']}")
    print(f"  • Hash Distance: {res_self['hash_distance']}")
    print(f"  • Is Visually Similar: {res_self['is_visually_similar']}")

    assert res_self["closest_match"] == "paypal_logo.png"
    assert res_self["hash_distance"] == 0
    print("PASS: Reference image matched itself with hash_distance = 0.")

    # Step 6: Test Completely Different Image (Poor Match)
    print("\n" + "=" * 80)
    print("6. TEST COMPLETELY DIFFERENT SYNTHETIC IMAGE (unrelated_forest_landscape.png)")
    print("=" * 80)
    diff_query = str(QUERY_DIR / "unrelated_forest_landscape.png")
    res_diff = compare_screenshot(
        image_path=diff_query,
        reference_dir=str(REFERENCE_DIR),
        threshold=10,
        precomputed_library=ref_lib,
    )
    print(f"Query Image: {Path(diff_query).name}")
    print(f"Result:")
    print(f"  • Closest Match: {res_diff['closest_match']}")
    print(f"  • Hash Distance: {res_diff['hash_distance']}")
    print(f"  • Is Visually Similar: {res_diff['is_visually_similar']}")
    print(f"  • All Distances: {res_diff['all_distances']}")

    assert res_diff["hash_distance"] is not None and res_diff["hash_distance"] > 10
    assert res_diff["is_visually_similar"] is False
    print(f"PASS: Poor match confirmed (hash_distance = {res_diff['hash_distance']} > threshold 10, is_visually_similar = False).")

    # Step 7: Test Error Handling and Robustness
    print("\n" + "=" * 80)
    print("7. TEST ERROR HANDLING AND ROBUSTNESS")
    print("=" * 80)

    # Missing image
    res_missing = compare_screenshot("non_existent_file.png", str(REFERENCE_DIR))
    print(f"Missing File Result: {res_missing}")
    assert res_missing["closest_match"] is None
    assert "error" in res_missing
    print("PASS: Handled non-existent image file gracefully.")

    # Corrupted image
    corrupt_query = str(QUERY_DIR / "corrupted_image.png")
    res_corrupt = compare_screenshot(corrupt_query, str(REFERENCE_DIR))
    print(f"Corrupt Image Result: {res_corrupt}")
    assert res_corrupt["closest_match"] is None
    assert "error" in res_corrupt
    print("PASS: Handled corrupted image file gracefully.")

    # Missing / empty reference directory
    res_empty_ref = compare_screenshot(clone_query, reference_dir="non_existent_ref_dir")
    print(f"Empty/Missing Reference Dir Result: {res_empty_ref}")
    assert res_empty_ref["closest_match"] is None
    assert res_empty_ref["hash_distance"] is None
    assert res_empty_ref["is_visually_similar"] is False
    print("PASS: Handled non-existent reference directory gracefully.")

    print("\n" + "=" * 80)
    print("ALL VISUAL BRAND SIMILARITY TESTS & ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
