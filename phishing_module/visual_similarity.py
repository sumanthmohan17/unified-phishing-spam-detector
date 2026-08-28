"""
Visual Brand Similarity Extractor
=================================
Module 1: Visual Impersonation Detection component for the Unified Phishing and
Email Spam Detection System.

Computes perceptual hashes (pHash) on webpage/logo screenshots and compares them
against a reference library of legitimate brand identities as described in
Section 3.2.1 and Section VII-C of the project report.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imagehash
from PIL import Image, UnidentifiedImageError

import tldextract

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}

# Known legitimate registered domain(s) for target monitored brands
KNOWN_BRAND_DOMAINS: Dict[str, List[str]] = {
    "paypal": ["paypal.com", "paypal.me"],
    "google": ["google.com", "google.co.in", "google.co.uk", "google.ca"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com", "msn.com"],
}


def extract_implied_brand(filename_or_path: str) -> str:
    """
    Extract the implied brand name from an image filename or path.

    Examples:
    - 'paypal_logo.png' -> 'paypal'
    - 'google-login-page.jpg' -> 'google'
    - 'microsoft.webp' -> 'microsoft'
    """
    stem = Path(filename_or_path).stem.lower()
    # Split by common delimiters: underscores, hyphens, spaces, dots
    parts = [p for p in re.split(r"[-_\s.]+", stem) if p]
    if not parts:
        return ""
    # The leading token is typically the brand name
    return parts[0]


def is_brand_matched(
    reference_brand: str,
    image_path: str,
    current_brand: Optional[str] = None,
    current_domain: Optional[str] = None,
) -> bool:
    """
    Determine if the current context (domain, declared brand, or filename)
    legitimately matches the reference brand identity.

    1. When current_domain is provided: Extracts the registered domain via tldextract
       and checks if it strictly matches one of the known domains in KNOWN_BRAND_DOMAINS.
       A brand name appearing as a token/substring in an untrusted domain (e.g.
       'paypal-secure-login.tk') is correctly rejected as unmatched/phishing.
    2. If ref_brand has no entry in KNOWN_BRAND_DOMAINS, falls back to False
       (unverified brand domain = suspicious).
    3. If current_domain is absent, infers from declared brand or query filename.
    """
    if not reference_brand:
        return False

    ref_brand = reference_brand.lower()

    # 1. Evaluate against current_domain using registered domain extraction
    if current_domain:
        extracted = tldextract.extract(current_domain.strip())
        reg_domain = (extracted.registered_domain or "").lower()

        known_domains = KNOWN_BRAND_DOMAINS.get(ref_brand, [])
        if not known_domains:
            # Unknown brand legitimacy -> treat as unmatched/suspicious
            return False

        # Must exactly match the registered domain of an authentic brand domain
        return reg_domain in [d.lower() for d in known_domains]

    # 2. Evaluate against current_brand if provided
    if current_brand:
        return ref_brand == current_brand.lower()

    # 3. If neither domain nor brand is provided, infer from image filename
    query_stem = Path(image_path).stem.lower()
    suspicious_tags = {"fake", "phish", "phishing", "clone", "spoof", "impersonation", "malicious", "attack"}
    query_tokens = set(re.split(r"[-_\s.]+", query_stem))

    if query_tokens & suspicious_tags:
        # File indicates a phishing/fake screenshot -> not legitimately matched
        return False

    if ref_brand in query_tokens:
        return True

    return False


def build_reference_library(source_dir: str) -> Dict[str, Any]:
    """
    Precompute and cache perceptual hashes for all images in a reference directory.

    Parameters
    ----------
    source_dir : str
        Directory path containing reference brand logos and page screenshots.

    Returns
    -------
    dict
        Dictionary mapping each image filename to its precomputed metadata:
        {
            "filename.png": {
                "hash": <imagehash.ImageHash>,
                "path": "/path/to/filename.png",
                "implied_brand": "brand_name"
            },
            ...
        }
    """
    if not source_dir or not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        return {}

    reference_library: Dict[str, Any] = {}

    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                img_full_path = os.path.join(root, fname)
                try:
                    with Image.open(img_full_path) as img:
                        # Compute 64-bit perceptual hash (pHash)
                        h = imagehash.phash(img)
                        brand = extract_implied_brand(fname)
                        reference_library[fname] = {
                            "hash": h,
                            "path": img_full_path,
                            "implied_brand": brand,
                        }
                except Exception:
                    # Ignore unreadable or corrupt reference images gracefully
                    continue

    return reference_library


def compare_screenshot(
    image_path: str,
    reference_dir: Optional[str] = None,
    threshold: int = 10,
    precomputed_library: Optional[Dict[str, Any]] = None,
    current_brand: Optional[str] = None,
    current_domain: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare a query screenshot against reference brand images using perceptual hashing.

    Parameters
    ----------
    image_path : str
        Path to the query image to inspect.
    reference_dir : Optional[str]
        Directory path containing reference images (used if precomputed_library is not provided).
    threshold : int, default=10
        Maximum Hamming distance for two perceptual hashes to be considered visually similar.
    precomputed_library : Optional[dict], default=None
        Cached dictionary returned by `build_reference_library`.
    current_brand : Optional[str], default=None
        Claimed or declared brand of the page/email.
    current_domain : Optional[str], default=None
        Domain name where the query image is hosted.

    Returns
    -------
    dict
        {
            "closest_match": Optional[str],
            "hash_distance": Optional[int],
            "is_visually_similar": bool,
            "all_distances": Dict[str, int],
            "error": Optional[str]  # Present only if an error occurred
        }
    """
    # 1. Validate and load query image
    if not image_path or not os.path.exists(image_path) or not os.path.isfile(image_path):
        return {
            "error": f"Image file not found: {image_path}",
            "closest_match": None,
            "hash_distance": None,
            "is_visually_similar": False,
            "all_distances": {},
        }

    try:
        with Image.open(image_path) as query_img:
            query_hash = imagehash.phash(query_img)
    except (UnidentifiedImageError, OSError, Exception) as exc:
        return {
            "error": f"Failed to open/decode image file '{image_path}': {exc}",
            "closest_match": None,
            "hash_distance": None,
            "is_visually_similar": False,
            "all_distances": {},
        }

    # 2. Acquire reference library
    if precomputed_library is not None:
        ref_lib = precomputed_library
    elif reference_dir:
        ref_lib = build_reference_library(reference_dir)
    else:
        ref_lib = {}

    if not ref_lib:
        return {
            "closest_match": None,
            "hash_distance": None,
            "is_visually_similar": False,
            "all_distances": {},
        }

    # 3. Compute Hamming distance against all reference hashes
    all_distances: Dict[str, int] = {}
    closest_match: Optional[str] = None
    min_distance: Optional[int] = None
    closest_brand: str = ""

    for fname, ref_entry in ref_lib.items():
        if isinstance(ref_entry, dict) and "hash" in ref_entry:
            ref_hash = ref_entry["hash"]
            brand = ref_entry.get("implied_brand", extract_implied_brand(fname))
        elif isinstance(ref_entry, imagehash.ImageHash):
            ref_hash = ref_entry
            brand = extract_implied_brand(fname)
        else:
            continue

        dist = int(query_hash - ref_hash)
        all_distances[fname] = dist

        if min_distance is None or dist < min_distance:
            min_distance = dist
            closest_match = fname
            closest_brand = brand

    # 4. Assess visual similarity and brand mismatch
    # is_visually_similar = True if hash_distance <= threshold BUT the context does not match the implied brand
    is_visually_similar = False
    if min_distance is not None and min_distance <= threshold and closest_match:
        matched = is_brand_matched(
            reference_brand=closest_brand,
            image_path=image_path,
            current_brand=current_brand,
            current_domain=current_domain,
        )
        # It looks like a known brand but isn't hosted as one -> Phishing impersonation signal!
        if not matched:
            is_visually_similar = True
        else:
            is_visually_similar = False

    return {
        "closest_match": closest_match,
        "hash_distance": min_distance,
        "is_visually_similar": is_visually_similar,
        "all_distances": all_distances,
    }
