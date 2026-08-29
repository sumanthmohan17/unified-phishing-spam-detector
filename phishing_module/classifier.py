"""
Multi-Signal Phishing Ensemble Classifier
=========================================
Module 1: Fused Ensemble Classifier for the Unified Phishing and Email Spam
Detection System.

Combines signals from:
1. URL Structural Analysis (feature_extraction.py)
2. Visual Brand Similarity (visual_similarity.py)
3. NLP Page Content Analysis (content_features.py)

Fuses all three feature vectors and classifies threats using a Stacking Ensemble
(Random Forest + XGBoost with a Random Forest meta-classifier) as described in
Section 3.2.1 and Section VIII-B of the project report.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .content_features import extract_content_features
from .feature_extraction import extract_url_features
from .visual_similarity import compare_screenshot

# Ordered list of fused numerical feature names used for model input
FEATURE_NAMES: List[str] = [
    "url_length",
    "subdomain_count",
    "has_ip_address",
    "uses_https",
    "hyphen_count",
    "special_char_count",
    "url_entropy",
    "domain_age_days",
    "has_redirect_chain",
    "is_visually_similar",
    "hash_distance",
    "urgency_score",
    "credential_request_score",
    "login_prompt_density",
    "overall_content_risk_score",
    "tfidf_term_count",
]


def build_feature_vector(
    url_features: Dict[str, Any],
    visual_features: Dict[str, Any],
    content_features: Dict[str, Any],
) -> Dict[str, float]:
    """
    Combine raw outputs from the three signal extractors into a single flat
    numeric feature vector suitable for machine learning model ingestion.

    Imputation and Representation Rationale:
    ----------------------------------------
    1. domain_age_days:
       - When None (due to raw IP host, WHOIS lookup failure, or timeout), imputed to 0.0.
       - Rationale: Freshly registered phishing domains often have an age of 0-7 days.
         Failed WHOIS lookups on suspicious or ephemeral domains correlate strongly with
         newly provisioned or unlisted malicious domains.
    2. hash_distance:
       - When None (missing screenshot or empty reference library), imputed to 64.0.
       - Rationale: 64 represents the maximum theoretical Hamming distance for 64-bit pHash,
         signifying maximal dissimilarity / absence of recognized brand match.
    3. Categorical / Non-numeric fields:
       - 'closest_match' and 'all_distances' are non-numeric strings/dicts and are omitted from
         direct model weights.
       - 'tfidf_top_terms' is represented numerically via 'tfidf_term_count' = len(tfidf_top_terms).
    4. Booleans:
       - Mapped to binary floats (1.0 for True, 0.0 for False).

    Parameters
    ----------
    url_features : dict
        Output from `extract_url_features`.
    visual_features : dict
        Output from `compare_screenshot`.
    content_features : dict
        Output from `extract_content_features`.

    Returns
    -------
    dict
        Flat dictionary of 16 numeric features.
    """
    # 1. URL Structural Features
    url_length = float(url_features.get("url_length", 0))
    subdomain_count = float(url_features.get("subdomain_count", 0))
    has_ip_address = 1.0 if url_features.get("has_ip_address") else 0.0
    uses_https = 1.0 if url_features.get("uses_https") else 0.0
    hyphen_count = float(url_features.get("hyphen_count", 0))
    special_char_count = float(url_features.get("special_char_count", 0))
    url_entropy = float(url_features.get("url_entropy", 0.0))

    raw_domain_age = url_features.get("domain_age_days")
    domain_age_days = float(raw_domain_age) if (raw_domain_age is not None and not np.isnan(raw_domain_age)) else 0.0

    has_redirect_chain = 1.0 if url_features.get("has_redirect_chain") else 0.0

    # 2. Visual Brand Similarity Features
    is_visually_similar = 1.0 if visual_features.get("is_visually_similar") else 0.0
    raw_hash_distance = visual_features.get("hash_distance")
    hash_distance = float(raw_hash_distance) if (raw_hash_distance is not None and not np.isnan(raw_hash_distance)) else 64.0

    # 3. NLP Page Content Features
    urgency_score = float(content_features.get("urgency_score", 0.0))
    credential_request_score = float(content_features.get("credential_request_score", 0.0))
    login_prompt_density = float(content_features.get("login_prompt_density", 0.0))
    overall_content_risk_score = float(content_features.get("overall_content_risk_score", 0.0))

    top_terms = content_features.get("tfidf_top_terms", [])
    tfidf_term_count = float(len(top_terms)) if isinstance(top_terms, list) else 0.0

    return {
        "url_length": url_length,
        "subdomain_count": subdomain_count,
        "has_ip_address": has_ip_address,
        "uses_https": uses_https,
        "hyphen_count": hyphen_count,
        "special_char_count": special_char_count,
        "url_entropy": url_entropy,
        "domain_age_days": domain_age_days,
        "has_redirect_chain": has_redirect_chain,
        "is_visually_similar": is_visually_similar,
        "hash_distance": hash_distance,
        "urgency_score": urgency_score,
        "credential_request_score": credential_request_score,
        "login_prompt_density": login_prompt_density,
        "overall_content_risk_score": overall_content_risk_score,
        "tfidf_term_count": tfidf_term_count,
    }


def train_ensemble(
    training_data: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[StackingClassifier, Dict[str, float]]:
    """
    Train a multi-signal Stacking Ensemble classifier per Section VIII-B.

    Stacking Ensemble Architecture:
    -------------------------------
    - Base Estimator 1: XGBClassifier (Gradient Boosted Trees - captures subtle non-linear
      interactions such as entropy + domain age).
    - Base Estimator 2: RandomForestClassifier (Bagged Trees - robust to noise and mixed signals).
    - Meta-Estimator: RandomForestClassifier combining base estimator prediction probabilities
      into the final unified threat probability.

    Parameters
    ----------
    training_data : pd.DataFrame
        DataFrame of feature vectors (matching FEATURE_NAMES).
    labels : pd.Series
        Binary target labels (1 for Phishing, 0 for Legitimate).
    test_size : float, default=0.2
        Fraction of dataset to hold out for validation metrics.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[StackingClassifier, dict]
        Trained stacking ensemble model and evaluation metrics dictionary:
        {"accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...}
    """
    X = training_data[FEATURE_NAMES]
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Base estimators
    xgb_base = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
    )
    rf_base = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=random_state,
    )

    # Meta-estimator combining stream outputs
    meta_rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=3,
        random_state=random_state,
    )

    # Stacking ensemble
    ensemble = StackingClassifier(
        estimators=[("xgb", xgb_base), ("rf", rf_base)],
        final_estimator=meta_rf,
        cv=3,
        n_jobs=1,
    )

    ensemble.fit(X_train, y_train)

    # Compute and store feature normalization statistics on the model for explainability
    feature_stats = {}
    for feat in FEATURE_NAMES:
        min_v = float(X_train[feat].min())
        max_v = float(X_train[feat].max())
        mean_v = float(X_train[feat].mean())
        std_v = float(X_train[feat].std()) if float(X_train[feat].std()) > 1e-6 else 1.0
        feature_stats[feat] = {
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "std": std_v,
        }
    ensemble.feature_stats_ = feature_stats

    # Compute evaluation metrics on held-out test split
    y_pred = ensemble.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    return ensemble, metrics


def train_on_real_data(
    n_phishing: int = 300,
    n_legitimate: int = 300,
    visual_reference_dir: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[StackingClassifier, dict]:
    """
    Fetch real phishing and legitimate URLs from the UCI PhiUSIIL Phishing URL
    Dataset (id=967), extract live features, and train the Stacking Ensemble
    classifier per Section 6.3 of the report.

    Parameters
    ----------
    n_phishing : int, default=300
        Target number of phishing URLs to sample from PhiUSIIL.
    n_legitimate : int, default=300
        Target number of legitimate URLs to sample from PhiUSIIL.
    visual_reference_dir : Optional[str]
        Optional path to reference images directory.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[StackingClassifier, dict]
        Trained model and empirical validation metrics dictionary.
    """
    from .real_data_loader import build_real_training_dataset, fetch_and_label_urls

    urls, labels = fetch_and_label_urls(
        n_phishing=n_phishing,
        n_legitimate=n_legitimate,
        random_state=random_state,
    )
    X, y = build_real_training_dataset(
        urls=urls,
        labels=labels,
        visual_reference_dir=visual_reference_dir,
    )
    model, metrics = train_ensemble(
        training_data=X,
        labels=y,
        random_state=random_state,
    )
    return model, metrics


def _extract_top_contributing_features(
    model: StackingClassifier,
    feature_vector: Dict[str, float],
    verdict: str = "phishing",
    top_k: int = 5,
) -> List[str]:
    """
    Identify the top features contributing to the prediction using the base
    estimators' learned feature importances weighted by min-max normalized
    feature activation values relative to training data distribution.
    """
    try:
        rf_model = model.named_estimators_["rf"]
        xgb_model = model.named_estimators_["xgb"]

        rf_imp = rf_model.feature_importances_
        xgb_imp = xgb_model.feature_importances_
        raw_imp = (rf_imp + xgb_imp) / 2.0

        # Apply additive smoothing to ensure binary signals are fairly weighted in instance attribution
        smoothed_imp = (raw_imp + 0.10) / (np.sum(raw_imp) + 0.10 * len(FEATURE_NAMES))

        feature_stats = getattr(model, "feature_stats_", None)

        contributions: List[Tuple[str, float]] = []
        for idx, feat_name in enumerate(FEATURE_NAMES):
            val = float(feature_vector.get(feat_name, 0.0))
            imp = float(smoothed_imp[idx])

            if feature_stats and feat_name in feature_stats:
                min_v = feature_stats[feat_name]["min"]
                max_v = feature_stats[feat_name]["max"]
                denom = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0
                norm_val = max(0.0, min(1.0, (val - min_v) / denom))
            else:
                norm_val = min(1.0, max(0.0, val))

            # Weight importance by the normalized feature activation for the verdict
            if verdict == "phishing":
                # For features where a lower value is risky (e.g. low domain age, no HTTPS)
                if feat_name in {"domain_age_days", "uses_https"}:
                    activation = 1.0 - norm_val
                elif feat_name == "hash_distance":
                    activation = (1.0 - norm_val) if feature_vector.get("is_visually_similar", 0.0) == 1.0 else 0.0
                else:
                    activation = norm_val

                # Binary red-flag features (has_ip_address, has_redirect_chain, is_visually_similar)
                # carry high diagnostic severity when actively triggered (= 1.0)
                if feat_name in {"has_ip_address", "has_redirect_chain", "is_visually_similar"} and val == 1.0:
                    activation = 1.25

                score = imp * activation
            else:
                # For legitimate verdict, higher domain age, HTTPS, low entropy, low risk scores
                if feat_name in {"domain_age_days", "uses_https"}:
                    activation = norm_val
                elif feat_name == "hash_distance":
                    activation = norm_val
                else:
                    activation = 1.0 - norm_val

                score = imp * activation

            contributions.append((feat_name, score))

        contributions.sort(key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in contributions[:top_k]]
    except Exception:
        # Fallback to key risk indicators if importance extraction fails
        return ["overall_content_risk_score", "has_ip_address", "is_visually_similar", "url_entropy"][:top_k]


def predict(
    url: str,
    screenshot_path: Optional[str],
    page_text: str,
    model: StackingClassifier,
    reference_dir: Optional[str] = None,
    precomputed_library: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run end-to-end multi-signal phishing prediction across all three extractors.

    Parameters
    ----------
    url : str
        The website URL to analyze.
    screenshot_path : Optional[str]
        Path to the rendered webpage/logo screenshot, or None if unavailable.
    page_text : str
        The extracted visible text from the webpage or email.
    model : StackingClassifier
        Trained ensemble classifier.
    reference_dir : Optional[str]
        Directory of reference brand images.
    precomputed_library : Optional[dict]
        Cached reference brand library.

    Returns
    -------
    dict
        {
            "verdict": "phishing" | "safe",
            "confidence": float (0.0 - 1.0),
            "phishing_probability": float (0.0 - 1.0),
            "top_features": List[str],
            "raw_scores": {
                "url_features": dict,
                "visual_features": dict,
                "content_features": dict,
            }
        }
    """
    # 1. Run Signal 1: URL Structural Feature Extraction
    url_features = extract_url_features(url)

    # 2. Run Signal 2: Visual Brand Similarity Extraction
    if screenshot_path and os.path.exists(screenshot_path):
        visual_features = compare_screenshot(
            image_path=screenshot_path,
            reference_dir=reference_dir,
            precomputed_library=precomputed_library,
            current_domain=url,
        )
    else:
        visual_features = {
            "closest_match": None,
            "hash_distance": None,
            "is_visually_similar": False,
            "all_distances": {},
        }

    # 3. Run Signal 3: NLP Page Content Feature Extraction
    content_features = extract_content_features(page_text)

    # 4. Fuse into single flat numeric feature vector
    feature_dict = build_feature_vector(url_features, visual_features, content_features)
    df_vector = pd.DataFrame([feature_dict])[FEATURE_NAMES]

    # 5. Model Inference
    proba_array = model.predict_proba(df_vector)[0]
    phishing_prob = float(proba_array[1])

    verdict = "phishing" if phishing_prob >= 0.5 else "safe"
    confidence = round(phishing_prob if verdict == "phishing" else (1.0 - phishing_prob), 4)

    # 6. Extract Explainable Top Contributing Features
    top_features = _extract_top_contributing_features(model, feature_dict, verdict=verdict, top_k=top_k)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "phishing_probability": round(phishing_prob, 4),
        "top_features": top_features,
        "raw_scores": {
            "url_features": url_features,
            "visual_features": visual_features,
            "content_features": content_features,
        },
    }


# ==============================================================================
# SYNTHETIC DATASET GENERATOR
# ==============================================================================
# NOTE: This dataset is synthetic and serves exclusively for unit testing and
# pipeline verification. It MUST be replaced with benchmark datasets
# (such as PhishTank and the UCI Machine Learning Phishing Dataset) before
# making real-world accuracy claims.
# ==============================================================================

SYNTHETIC_TRAINING_DATA_NOTICE = (
    "SYNTHETIC_TRAINING_DATA: Used for testing pipeline execution and verification. "
    "Must be replaced with PhishTank and UCI ML Phishing Dataset before real-world claims."
)


def generate_synthetic_training_data(
    n_samples: int = 100,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic multi-signal training data (~100 samples) across diverse
    phishing and legitimate profiles for pipeline verification.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of samples (roughly 50% phishing, 50% legitimate).
    random_state : int, default=42
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature DataFrame (matching FEATURE_NAMES) and binary labels (1=Phishing, 0=Legitimate).
    """
    np.random.seed(random_state)
    records = []
    labels = []

    n_phish = n_samples // 2
    n_legit = n_samples - n_phish

    # --- Generate Synthetic Phishing Profiles (Label = 1) ---
    for _ in range(n_phish):
        url_len = np.random.randint(45, 120)
        subdomains = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        is_ip = 1.0 if np.random.rand() < 0.35 else 0.0
        uses_https = 1.0 if np.random.rand() < 0.60 else 0.0
        hyphens = float(np.random.randint(1, 5))
        special_chars = float(np.random.randint(7, 20))
        entropy = round(float(np.random.uniform(3.9, 5.2)), 4)
        domain_age = float(np.random.choice([0, np.random.randint(1, 30)]))
        redirect = 1.0 if np.random.rand() < 0.40 else 0.0

        is_vis_sim = 1.0 if np.random.rand() < 0.65 else 0.0
        hash_dist = float(np.random.randint(0, 10)) if is_vis_sim else float(np.random.randint(20, 64))

        urgency = round(float(np.random.uniform(0.4, 0.95)), 4)
        credential = round(float(np.random.uniform(0.3, 0.90)), 4)
        login_density = round(float(np.random.uniform(0.2, 0.60)), 4)
        overall_risk = round(0.45 * urgency + 0.35 * credential + 0.20 * login_density, 4)
        tfidf_count = float(np.random.randint(6, 10))

        row = {
            "url_length": float(url_len),
            "subdomain_count": float(subdomains),
            "has_ip_address": is_ip,
            "uses_https": uses_https,
            "hyphen_count": hyphens,
            "special_char_count": special_chars,
            "url_entropy": entropy,
            "domain_age_days": domain_age,
            "has_redirect_chain": redirect,
            "is_visually_similar": is_vis_sim,
            "hash_distance": hash_dist,
            "urgency_score": urgency,
            "credential_request_score": credential,
            "login_prompt_density": login_density,
            "overall_content_risk_score": overall_risk,
            "tfidf_term_count": tfidf_count,
        }
        records.append(row)
        labels.append(1)

    # --- Generate Synthetic Legitimate Profiles (Label = 0) ---
    for _ in range(n_legit):
        url_len = np.random.randint(15, 45)
        subdomains = np.random.choice([0, 1], p=[0.7, 0.3])
        is_ip = 0.0
        uses_https = 1.0 if np.random.rand() < 0.92 else 0.0
        hyphens = float(np.random.choice([0, 1], p=[0.85, 0.15]))
        special_chars = float(np.random.randint(3, 7))
        entropy = round(float(np.random.uniform(2.8, 3.8)), 4)
        domain_age = float(np.random.randint(365, 10000))
        redirect = 0.0

        is_vis_sim = 0.0
        hash_dist = float(np.random.randint(25, 64))

        urgency = 0.0 if np.random.rand() < 0.85 else round(float(np.random.uniform(0.0, 0.15)), 4)
        credential = 0.0 if np.random.rand() < 0.80 else round(float(np.random.uniform(0.0, 0.20)), 4)
        login_density = 0.0 if np.random.rand() < 0.70 else round(float(np.random.uniform(0.0, 0.25)), 4)
        overall_risk = round(0.45 * urgency + 0.35 * credential + 0.20 * login_density, 4)
        tfidf_count = float(np.random.randint(5, 10))

        row = {
            "url_length": float(url_len),
            "subdomain_count": float(subdomains),
            "has_ip_address": is_ip,
            "uses_https": uses_https,
            "hyphen_count": hyphens,
            "special_char_count": special_chars,
            "url_entropy": entropy,
            "domain_age_days": domain_age,
            "has_redirect_chain": redirect,
            "is_visually_similar": is_vis_sim,
            "hash_distance": hash_dist,
            "urgency_score": urgency,
            "credential_request_score": credential,
            "login_prompt_density": login_density,
            "overall_content_risk_score": overall_risk,
            "tfidf_term_count": tfidf_count,
        }
        records.append(row)
        labels.append(0)

    # Shuffle dataset
    indices = np.arange(len(records))
    np.random.shuffle(indices)

    df_shuffled = pd.DataFrame([records[i] for i in indices])
    y_shuffled = pd.Series([labels[i] for i in indices])

    return df_shuffled, y_shuffled
