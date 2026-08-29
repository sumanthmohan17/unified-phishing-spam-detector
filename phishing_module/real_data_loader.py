"""
Real-World Phishing & Legitimate Benchmark Dataset Loader
=========================================================
Module 1 Data Loader: Sourcing real phishing URLs from OpenPhish and
legitimate URLs from Tranco Top Sites, extracting live features for
empirical ensemble model training as described in Section 6.3 of the report.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple


import pandas as pd
import requests

from .classifier import build_feature_vector
from .feature_extraction import extract_url_features

logger = logging.getLogger("phishing_module.data_loader")

CACHE_DIR = Path(__file__).parent / "data"
CACHE_FILE = CACHE_DIR / "phiusiil_urls.parquet"


def fetch_and_label_urls(
    n_phishing: int = 300,
    n_legitimate: int = 300,
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Fetch real phishing and legitimate URLs from the UCI PhiUSIIL Phishing URL
    Dataset (id=967) via ucimlrepo.

    In the PhiUSIIL dataset:
      - label == 0: Phishing (~101k URLs) -> mapped to our label 1
      - label == 1: Legitimate (~135k URLs) -> mapped to our label 0

    Both classes are full realistic URLs sampled under uniform collection
    methodology, avoiding path/length distribution bias.

    Parameters
    ----------
    n_phishing : int, default=300
        Number of phishing URLs to sample.
    n_legitimate : int, default=300
        Number of legitimate URLs to sample.
    random_state : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    tuple[list[str], list[int]]
        (urls, labels) where 1 indicates phishing and 0 indicates legitimate.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df_urls: Optional[pd.DataFrame] = None

    # Check local cache first for fast repeat runs
    if CACHE_FILE.exists():
        try:
            logger.info(f"Loading cached PhiUSIIL raw URLs from {CACHE_FILE}...")
            df_urls = pd.read_parquet(CACHE_FILE)
            logger.info(f"Loaded {len(df_urls)} URLs from cache.")
        except Exception as exc:
            logger.warning(f"Failed to read cache file: {exc}. Re-fetching from UCI repository...")
            df_urls = None

    if df_urls is None or df_urls.empty:
        logger.info("Fetching PhiUSIIL Phishing URL Dataset (id=967) via ucimlrepo...")
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=967)
        if hasattr(dataset.data, "original") and dataset.data.original is not None:
            raw_df = dataset.data.original[["URL", "label"]].copy()
        else:
            raw_df = pd.concat([dataset.data.features["URL"], dataset.data.targets["label"]], axis=1)

        # Clean string URLs
        raw_df["URL"] = raw_df["URL"].astype(str).str.strip()
        raw_df = raw_df.dropna(subset=["URL", "label"])
        raw_df = raw_df[raw_df["URL"].str.len() > 0]
        df_urls = raw_df

        # Save to local cache
        try:
            df_urls.to_parquet(CACHE_FILE, index=False)
            logger.info(f"Cached {len(df_urls)} PhiUSIIL raw URLs to {CACHE_FILE}.")
        except Exception as exc:
            logger.warning(f"Could not cache to parquet: {exc}")

    # PhiUSIIL: label == 0 is Phishing, label == 1 is Legitimate
    phishing_subset = df_urls[df_urls["label"] == 0]["URL"]
    legitimate_subset = df_urls[df_urls["label"] == 1]["URL"]

    # Sample required counts
    phishing_sample = phishing_subset.sample(n=min(n_phishing, len(phishing_subset)), random_state=random_state).tolist()
    legitimate_sample = legitimate_subset.sample(n=min(n_legitimate, len(legitimate_subset)), random_state=random_state).tolist()

    logger.info(f"Sampled {len(phishing_sample)} phishing URLs (target label=1) and {len(legitimate_sample)} legitimate URLs (target label=0).")

    # Combine into unified dataset: Phishing -> 1, Legitimate -> 0
    combined_urls = phishing_sample + legitimate_sample
    combined_labels = [1] * len(phishing_sample) + [0] * len(legitimate_sample)

    return combined_urls, combined_labels


def build_real_training_dataset(
    urls: List[str],
    labels: List[int],
    visual_reference_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract real-world structural features for a list of URLs and fuse them into
    a machine learning feature matrix.

    Note on URL-Only Benchmark Dataset:
    -----------------------------------
    For large-scale URL benchmark datasets (such as OpenPhish + Tranco), live rendered
    screenshots and parsed web page bodies are not included. Neutral/default values are
    used for visual (hash_distance=64.0, is_visually_similar=False) and content signals
    (zeroed NLP scores) to avoid fabricating artificial data. This trains the ensemble
    primarily on robust URL structural patterns, matching the URL-only browser extension
    evaluation pipeline.

    Parameters
    ----------
    urls : list[str]
        List of URL strings to process.
    labels : list[int]
        Corresponding binary class labels (1=phishing, 0=legitimate).
    visual_reference_dir : Optional[str]
        Optional path to brand reference logo library.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X is the 16-feature DataFrame and y is the binary labels Series.
    """
    records: List[dict] = []
    valid_labels: List[int] = []
    total = len(urls)

    # Neutral visual and content feature baselines for URL-only benchmark dataset
    neutral_visual_features = {
        "closest_match": None,
        "hash_distance": 64.0,
        "is_visually_similar": False,
        "all_distances": {},
    }
    neutral_content_features = {
        "urgency_score": 0.0,
        "credential_request_score": 0.0,
        "login_prompt_density": 0.0,
        "tfidf_top_terms": [],
        "overall_content_risk_score": 0.0,
    }

    t0 = time.time()
    logger.info(f"Starting real feature extraction across {total} URLs...")

    for idx, (url, label) in enumerate(zip(urls, labels), start=1):
        try:
            # 1. Live structural feature extraction (includes WHOIS lookup, entropy, etc.)
            url_feats = extract_url_features(url)

            # 2. Fuse into unified 16-dimensional feature vector
            vector = build_feature_vector(
                url_features=url_feats,
                visual_features=neutral_visual_features,
                content_features=neutral_content_features,
            )

            records.append(vector)
            valid_labels.append(label)
        except Exception as exc:
            logger.warning(f"Error extracting features for URL '{url}': {exc}")
            continue

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t0
            avg_per_url = elapsed / idx if idx > 0 else 0
            print(
                f"  [Feature Extraction] Processed {idx}/{total} URLs "
                f"({(idx/total)*100:.1f}%) | Elapsed: {elapsed:.1f}s ({avg_per_url:.2f}s/URL)"
            )

    X = pd.DataFrame(records)
    y = pd.Series(valid_labels, name="label")

    logger.info(f"Feature extraction completed for {len(X)} URLs in {time.time() - t0:.2f}s.")
    return X, y
