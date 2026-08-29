"""
Email Spam Classification & Phishing Integration Classifier
===========================================================
Module 2: 4-Category Email Classifier (Legitimate, Spam, Phishing, Malware)
fused with Module 1 (Phishing Detection) for automatic embedded URL routing
as described in Section 3.2.2 and Section VIII-C of the project report.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Direct import and routing into Module 1
import phishing_module

from .preprocessing import preprocess_email_text
from .url_extraction import extract_urls_from_email

LABEL_MAPPING: Dict[int, str] = {
    0: "legitimate",
    1: "spam",
    2: "phishing",
    3: "malware",
}

NAME_TO_LABEL: Dict[str, int] = {v: k for k, v in LABEL_MAPPING.items()}

# Promotional & spam vocabulary keywords
SPAM_PROMO_KEYWORDS: List[str] = [
    "free", "guaranteed", "winner", "cash", "discount", "buy direct",
    "order now", "unlimited", "casino", "lottery", "bonus", "risk free",
    "special promotion", "click here", "save big", "exclusive deal",
    "lowest price", "weight loss", "viagra", "cheap", "prizes", "income",
]

# Malware & malicious attachment vocabulary keywords
MALWARE_KEYWORDS: List[str] = [
    "invoice attached", "executable", "zip archive", "macro enabled",
    "enable macros", "run script", "powershell", "payload", "trojan",
    "downloader", "extract zip", "encrypted attachment", "receipt attached",
    "view document", "iso image", "vbs script", "malicious payload",
]

# Phishing urgency & credential harvesting vocabulary keywords
PHISHING_KEYWORDS: List[str] = [
    "account suspended", "verify password", "confirm identity", "security alert",
    "within 24 hours", "banking update", "login credentials", "unauthorized access",
    "billing problem", "immediate action required", "locked account", "update payment",
]


def extract_spam_features(email_text: str) -> Dict[str, float]:
    """
    Extract linguistic and threat-density features from cleaned email text.

    Parameters
    ----------
    email_text : str
        Cleaned/preprocessed email text string.

    Returns
    -------
    dict
        Extracted numerical features.
    """
    if not email_text or not isinstance(email_text, str):
        return {
            "spam_promo_density": 0.0,
            "malware_indicator_density": 0.0,
            "phishing_urgency_density": 0.0,
            "word_count": 0.0,
        }

    text_lower = email_text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text_lower)
    n_tokens = max(len(tokens), 1)

    def _calc_density(phrases: List[str]) -> float:
        count = 0
        for p in phrases:
            matches = len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
            count += matches * len(p.split())
        return min(1.0, round((count / n_tokens) * 3.0, 4))

    return {
        "spam_promo_density": _calc_density(SPAM_PROMO_KEYWORDS),
        "malware_indicator_density": _calc_density(MALWARE_KEYWORDS),
        "phishing_urgency_density": _calc_density(PHISHING_KEYWORDS),
        "word_count": float(len(tokens)),
    }


class SpamStackingPipeline(BaseEstimator, ClassifierMixin):
    """
    Unified Stacking Pipeline for Email Classification:
    Combines TF-IDF Bag-of-Words + Dense Engineered Threat Features
    into Naive Bayes & XGBoost base learners with a Random Forest meta-classifier.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=250,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.base_nb = MultinomialNB(alpha=0.1)
        self.base_xgb = XGBClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=4,
        )
        self.meta_rf = RandomForestClassifier(
            n_estimators=40,
            max_depth=3,
            random_state=random_state,
        )
        self.stacking_ensemble: Optional[StackingClassifier] = None

    def _build_dense_features(self, raw_texts: List[str]) -> np.ndarray:
        tfidf_mat = self.vectorizer.transform(raw_texts).toarray()
        dense_extra = []
        for text in raw_texts:
            feats = extract_spam_features(text)
            dense_extra.append([
                feats["spam_promo_density"],
                feats["malware_indicator_density"],
                feats["phishing_urgency_density"],
                min(1.0, feats["word_count"] / 100.0),
            ])
        combined = np.hstack([tfidf_mat, np.array(dense_extra)])
        return combined

    def fit(self, raw_texts: List[str], y: np.ndarray):
        # 1. Fit TF-IDF on training texts
        self.vectorizer.fit(raw_texts)
        X_combined = self._build_dense_features(raw_texts)

        # 2. Build and fit stacking ensemble
        self.stacking_ensemble = StackingClassifier(
            estimators=[
                ("nb", self.base_nb),
                ("xgb", self.base_xgb),
            ],
            final_estimator=self.meta_rf,
            cv=3,
            n_jobs=1,
        )
        self.stacking_ensemble.fit(X_combined, y)
        return self

    def predict(self, raw_texts: List[str]) -> np.ndarray:
        X_combined = self._build_dense_features(raw_texts)
        return self.stacking_ensemble.predict(X_combined)

    def predict_proba(self, raw_texts: List[str]) -> np.ndarray:
        X_combined = self._build_dense_features(raw_texts)
        return self.stacking_ensemble.predict_proba(X_combined)


def train_spam_ensemble(
    training_data: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[SpamStackingPipeline, Dict[str, Any]]:
    """
    Train the 4-Class Email Spam Classification Ensemble per Section VIII-C.

    Categories:
    0: Legitimate
    1: Spam
    2: Phishing
    3: Malware

    Parameters
    ----------
    training_data : pd.DataFrame
        DataFrame with 'email_text' column.
    labels : pd.Series
        Integer or string labels (0/legitimate, 1/spam, 2/phishing, 3/malware).
    test_size : float, default=0.2
        Validation split fraction.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    tuple[SpamStackingPipeline, dict]
        Trained model pipeline and evaluation metrics dictionary.
    """
    texts = training_data["email_text"].tolist()
    # Normalize labels to integers (0, 1, 2, 3)
    if isinstance(labels.iloc[0], str):
        y = np.array([NAME_TO_LABEL[label.lower()] for label in labels])
    else:
        y = np.array(labels.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = SpamStackingPipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Compute overall and per-class metrics
    acc = round(float(accuracy_score(y_test, y_pred)), 4)
    prec_macro = round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    rec_macro = round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    f1_macro = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)

    per_class = {}
    for class_idx, class_name in LABEL_MAPPING.items():
        mask = (y_test == class_idx)
        if np.sum(mask) > 0:
            p = precision_score(y_test == class_idx, y_pred == class_idx, zero_division=0)
            r = recall_score(y_test == class_idx, y_pred == class_idx, zero_division=0)
            f = f1_score(y_test == class_idx, y_pred == class_idx, zero_division=0)
            per_class[class_name] = {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f), 4),
            }

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "per_class_metrics": per_class,
    }

    return pipeline, metrics


def classify_email(
    email_body: str,
    model: SpamStackingPipeline,
    phishing_reference_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify an email with automatic URL extraction and live routing into Module 1.

    Core Integration Pipeline (Section VIII-C):
    -------------------------------------------
    1. Preprocesses the email body text (HTML removal, tokenization, stop words).
    2. Extracts all embedded URLs using regex pattern matching.
    3. For EACH embedded URL, calls `phishing_module.extract_url_features()` to obtain
       live structural threat indicators (IP hostname, redirect shorteners, entropy, etc.).
       (Note: URL structural signals are evaluated here; screenshot-based visual analysis is
       omitted because emails deliver hyperlinks rather than rendered visual page screenshots).
    4. Computes text classification probabilities (Legitimate, Spam, Phishing, Malware).
    5. Fuses the email text classification with embedded URL threat signals:
       - If any embedded URL is flagged as high-risk phishing by Module 1 (e.g. IP host or
         shortener redirect chain) OR model predicts phishing -> verdict: "phishing".
       - Else if model predicts malware -> verdict: "malware".
       - Else if model predicts spam -> verdict: "spam".
       - Otherwise -> verdict: "legitimate".

    Parameters
    ----------
    email_body : str
        Raw email body content (plain text or HTML).
    model : SpamStackingPipeline
        Trained email spam stacking classifier.
    phishing_reference_dir : Optional[str]
        Optional reference logos directory for visual matching.

    Returns
    -------
    dict
        {
            "verdict": "legitimate" | "spam" | "phishing" | "malware",
            "confidence": float (0.0 - 1.0),
            "embedded_urls": List[str],
            "embedded_url_risk": Dict[str, dict], # per-URL phishing features from Module 1
            "top_features": List[str],
        }
    """
    # 1. Preprocess email text
    cleaned_text = preprocess_email_text(email_body)

    # 2. Extract embedded URLs from raw email body
    embedded_urls = extract_urls_from_email(email_body)

    # 3. ROUTE EMBEDDED URLS INTO MODULE 1 (Phishing Detection)
    embedded_url_risk: Dict[str, Any] = {}
    has_malicious_phishing_url = False
    url_threat_flags: List[str] = []

    for url in embedded_urls:
        try:
            # Genuine function call into Module 1
            url_feats = phishing_module.extract_url_features(url)
            embedded_url_risk[url] = url_feats

            # Check for critical phishing red-flags from Module 1
            if url_feats.get("has_ip_address"):
                has_malicious_phishing_url = True
                url_threat_flags.append(f"IP-based URL ({url})")
            if url_feats.get("has_redirect_chain"):
                has_malicious_phishing_url = True
                url_threat_flags.append(f"Shortener redirect chain ({url})")
            if url_feats.get("url_entropy", 0.0) >= 4.5:
                has_malicious_phishing_url = True
                url_threat_flags.append(f"High-entropy obfuscated URL ({url})")
        except Exception:
            # If an individual URL is malformed, record error gracefully
            embedded_url_risk[url] = {"error": "Failed to parse URL structural features"}

    # 4. Text Classification via Stacking Model
    if not cleaned_text:
        # Fallback for empty text
        text_probs = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        text_probs = model.predict_proba([cleaned_text])[0]

    prob_legit, prob_spam, prob_phish, prob_malware = text_probs

    # 5. Fused Decision Logic
    top_features: List[str] = []
    text_feats = extract_spam_features(cleaned_text)

    if has_malicious_phishing_url or prob_phish >= max(prob_legit, prob_spam, prob_malware):
        verdict = "phishing"
        confidence = round(float(max(prob_phish, 0.92 if has_malicious_phishing_url else 0.50)), 4)
        top_features.extend(url_threat_flags)
        if text_feats["phishing_urgency_density"] > 0:
            top_features.append("phishing_urgency_density")
        if prob_phish > 0.3:
            top_features.append("nlp_phishing_intent")
    elif prob_malware >= max(prob_legit, prob_spam):
        verdict = "malware"
        confidence = round(float(prob_malware), 4)
        if text_feats["malware_indicator_density"] > 0:
            top_features.append("malware_indicator_density")
        top_features.append("malicious_attachment_patterns")
    elif prob_spam >= prob_legit:
        verdict = "spam"
        confidence = round(float(prob_spam), 4)
        if text_feats["spam_promo_density"] > 0:
            top_features.append("spam_promo_density")
        top_features.append("unsolicited_commercial_content")
    else:
        verdict = "legitimate"
        confidence = round(float(prob_legit), 4)
        top_features.append("authentic_communication_style")

    if not top_features:
        top_features.append("model_class_probability")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "embedded_urls": embedded_urls,
        "embedded_url_risk": embedded_url_risk,
        "top_features": top_features[:5],
    }


# ==============================================================================
# SYNTHETIC DATASET GENERATOR FOR MODULE 2
# ==============================================================================
# NOTE: This dataset is synthetic and serves exclusively for unit testing and
# pipeline verification. It MUST be replaced with benchmark corpora (such as
# Enron Email Dataset and SpamAssassin) before making real-world accuracy claims.
# ==============================================================================

SYNTHETIC_TRAINING_DATA_NOTICE = (
    "SYNTHETIC_TRAINING_DATA: Used for testing Module 2 execution and URL routing. "
    "Must be replaced with Enron / SpamAssassin corpus for real-world claims."
)


def generate_synthetic_spam_data(
    n_samples: int = 100,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic email training samples (~100 samples) balanced across
    all 4 target categories (Legitimate, Spam, Phishing, Malware).
    """
    np.random.seed(random_state)
    records = []
    labels = []

    samples_per_class = n_samples // 4

    # 1. Legitimate Emails (Label = 0)
    legit_templates = [
        "Hi Team, attached is the revised project agenda for tomorrow's sprint planning meeting. Please review the action items.",
        "Good morning, could you send me the updated quarterly financial report? Thanks, Sarah.",
        "Dear customer, your scheduled service appointment for Friday at 10 AM is confirmed. Call our office if you need to reschedule.",
        "Reminder: The company-wide engineering seminar starts at 2 PM in Conference Room B. Refreshments will be provided.",
        "Hi John, thanks for your feedback on the API design proposal. I have addressed your comments in the new pull request.",
    ]

    # 2. Spam Promotional Emails (Label = 1)
    spam_templates = [
        "Congratulations winner! You have been selected for a free $1,000 cash bonus. Claim your prize now, no purchase necessary!",
        "Special exclusive promotion: Get 80% discount on all luxury watches and jewelry. Buy direct today and save big!",
        "Risk free investment opportunity! Unlimited income potential from home. Click here to enroll in our free crypto trading course.",
        "Special pharmacy deals! Lowest prices guaranteed on weight loss medications and supplements. Order today for free shipping.",
        "Exclusive casino bonus: Claim your 500 free spins and double your first deposit instantly. Play now to win massive jackpots!",
    ]

    # 3. Phishing Credential Emails (Label = 2)
    phish_templates = [
        "Critical Security Alert: Your PayPal account has been suspended due to unauthorized access. Confirm your password immediately: http://192.168.1.55/login",
        "Urgent: Microsoft 365 password expires in 24 hours. Sign in to verify your account credentials: https://paypa1-verify.tk/auth",
        "Bank Security Notice: Unusual activity detected on your credit card. Update your banking credentials within 12 hours: http://bit.ly/bank-auth",
        "Apple ID Warning: Your iCloud storage has been locked. Verify billing details immediately to restore access: http://10.0.0.1/verify.php",
        "Immediate Action Required: Unauthorized login from Russia. Click here to confirm your identity and prevent permanent account termination: http://tinyurl.com/auth-id",
    ]

    # 4. Malware & Malicious Attachment Emails (Label = 3)
    malware_templates = [
        "Please find your overdue invoice attached in the zip archive. Enable macros to view the invoice document details.",
        "Shipping notice: Your package delivery failed. Extract the attached zip archive and run the tracking executable.",
        "Urgent court subpoena notice: Download and execute the attached payroll document to view official legal proceedings.",
        "Purchase Order confirmation: Encrypted invoice payload attached. Run script or enable macros in Microsoft Office to decrypt.",
        "Scanned document from Xerox printer: Open the attached ISO image file and run setup to inspect the electronic receipt.",
    ]

    for template in legit_templates:
        for _ in range(samples_per_class // len(legit_templates)):
            records.append(template + f" ref-{np.random.randint(100, 999)}")
            labels.append(0)

    for template in spam_templates:
        for _ in range(samples_per_class // len(spam_templates)):
            records.append(template + f" promo-{np.random.randint(100, 999)}")
            labels.append(1)

    for template in phish_templates:
        for _ in range(samples_per_class // len(phish_templates)):
            records.append(template + f" alert-{np.random.randint(100, 999)}")
            labels.append(2)

    for template in malware_templates:
        for _ in range(samples_per_class // len(malware_templates)):
            records.append(template + f" doc-{np.random.randint(100, 999)}")
            labels.append(3)

    # Fill remainder if any
    while len(records) < n_samples:
        records.append("Legitimate project update email for team synchronization.")
        labels.append(0)

    idx = np.arange(len(records))
    np.random.shuffle(idx)

    df = pd.DataFrame({"email_text": [records[i] for i in idx]})
    y = pd.Series([labels[i] for i in idx])

    return df, y
