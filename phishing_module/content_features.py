"""
NLP Page Content Analysis Extractor
===================================
Module 1: NLP Content Feature Extraction component for the Unified Phishing and
Email Spam Detection System.

Analyzes raw webpage/email text for psychological urgency, credential requests,
login form indicators, and TF-IDF top terms as described in Section 3.2.1
("NLP Page Content Analysis") of the project report.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Module-level Constants for Psychological & Content Analysis

URGENCY_PHRASES: List[str] = [
    "immediately",
    "urgent action required",
    "urgent",
    "act now",
    "will be suspended",
    "account suspended",
    "suspended immediately",
    "expires today",
    "within 24 hours",
    "within 48 hours",
    "24 hours",
    "48 hours",
    "immediate verification required",
    "unauthorized access detected",
    "terminate your account",
    "restricted access",
    "security alert",
    "locked out",
    "limited time",
    "action required",
    "respond immediately",
    "final notice",
    "critical security notice",
    "verify immediately",
    "unusual activity detected",
    "permanent closure",
    "failure to respond",
    "temporary suspension",
    "deadline",
    "account will be locked",
    "immediate action",
]

CREDENTIAL_PHRASES: List[str] = [
    "enter your password",
    "confirm your password",
    "verify your password",
    "verify your account",
    "confirm your account",
    "update your billing",
    "update payment details",
    "enter your pin",
    "security questions",
    "social security number",
    "ssn",
    "credit card details",
    "banking credentials",
    "card number",
    "cvv",
    "confirm identity",
    "re-enter password",
    "provide your credentials",
    "account credentials",
    "login credentials",
    "unlock account",
    "validate your account",
    "sign in to confirm",
    "enter verification code",
    "submit your credentials",
    "verify billing information",
    "security passcode",
]

LOGIN_PROMPT_TERMS: List[str] = [
    "username",
    "password",
    "sign in",
    "signin",
    "log in",
    "login",
    "verify account",
    "confirm identity",
    "enter email",
    "user id",
    "passcode",
    "remember me",
    "forgot password",
    "submit",
    "authenticate",
    "two-factor authentication",
    "2fa",
    "otp",
    "security pin",
]

# 2. Built-in Reference Corpus for Meaningful TF-IDF IDF Weighting
REFERENCE_CORPUS: List[str] = [
    # --- 10 Representative Phishing Samples ---
    "Urgent: Your PayPal account has been restricted due to unauthorized login attempts. Confirm your password immediately to avoid suspension.",
    "Critical security alert: Microsoft 365 password expires today. Please sign in and verify your credentials within 24 hours.",
    "Bank of America Security Notice: Suspicious transactions detected on your debit card. Enter your PIN and verify account details now.",
    "Google Account Warning: Someone in Russia attempted to access your Gmail. Verify your identity and update security questions immediately.",
    "Netflix Billing Failure: We could not process your subscription payment. Update your credit card details and CVV to restore access.",
    "Apple ID Locked: Your account has been temporarily disabled for security reasons. Click here to confirm your account and unlock access.",
    "Amazon Security Notification: Unauthorized order placed for iPhone 15. Sign in to confirm or cancel this charge within 24 hours.",
    "Internal Revenue Service Tax Refund: You have an unclaimed tax refund of $1,250. Enter your SSN and banking credentials to claim.",
    "Cloud Storage Quota Exceeded: Your mailbox will be deactivated in 12 hours. Login with your username and password to upgrade storage.",
    "Meta Account Termination Notice: Your Facebook business page violates copyright policy. Submit your login credentials to appeal immediately.",

    # --- 10 Representative Legitimate Samples ---
    "Welcome to Acme Corp! We manufacture high-efficiency thermal insulation materials and building solutions for commercial architects.",
    "Product Overview: The Wireless Precision Mouse features an ergonomic contour, 2.4GHz USB receiver, and up to 18 months of battery life.",
    "University Computer Science Course Syllabus: Introduction to Data Structures and Algorithms covering binary trees, sorting, and Big-O notation.",
    "Classic Chocolate Chip Cookie Recipe: Cream together softened butter, brown sugar, and vanilla extract before folding in semi-sweet chocolate chips.",
    "Open Source Project Documentation: Learn how to configure Docker containers and orchestrate microservices with Kubernetes in production.",
    "Privacy Policy: We value your trust and do not sell your personal information to third-party advertisers without explicit consent.",
    "Cloud Computing Insights: Exploring architectural best practices for deploying resilient, multi-region serverless applications on AWS.",
    "Customer Support FAQ: Standard shipping takes 3-5 business days across the continental US, and tracking numbers are sent via confirmation email.",
    "Quarterly Financial Summary: The company reported strong revenue growth in the enterprise cloud sector with operating margins expanding by 4%.",
    "Password Reset Notice: A password reset request was received for your account. If you initiated this, click the link to set a new password.",
]


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


VERB_INFLECTIONS: Dict[str, str] = {
    "confirm": r"(?:confirm|confirming|confirmed|confirms)",
    "verify": r"(?:verify|verifying|verified|verifies|verification)",
    "enter": r"(?:enter|entering|entered|enters)",
    "update": r"(?:update|updating|updated|updates)",
    "submit": r"(?:submit|submitting|submitted|submits)",
    "provide": r"(?:provide|providing|provided|provides)",
    "suspend": r"(?:suspend|suspending|suspended|suspends|suspension)",
    "restrict": r"(?:restrict|restricting|restricted|restricts|restriction)",
    "terminate": r"(?:terminate|terminating|terminated|terminates|termination)",
    "lock": r"(?:lock|locking|locked|locks)",
    "expire": r"(?:expire|expiring|expired|expires|expiration)",
    "require": r"(?:require|requiring|required|requires)",
    "respond": r"(?:respond|responding|responded|responds|response)",
}


def _compile_phrase_regex(phrase: str) -> re.Pattern:
    """Compile a phrase into a regex pattern supporting inflectional variants."""
    tokens = phrase.lower().strip().split()
    pattern_parts = []
    for token in tokens:
        if token in VERB_INFLECTIONS:
            pattern_parts.append(VERB_INFLECTIONS[token])
        else:
            pattern_parts.append(re.escape(token))
    full_pattern = r"\b" + r"\s+".join(pattern_parts) + r"\b"
    return re.compile(full_pattern, re.IGNORECASE)


def calculate_phrase_density(text: str, phrases: List[str], tokens: List[str]) -> float:
    """
    Calculate normalized density score (0.0 - 1.0) of target phrases within a text.
    """
    if not text or not tokens or len(tokens) < 3:
        return 0.0

    text_lower = text.lower()
    total_tokens = len(tokens)
    matched_token_count = 0

    for phrase in phrases:
        phrase_clean = phrase.lower().strip()
        if not phrase_clean:
            continue
        # Use inflection-aware regex matching
        pattern = _compile_phrase_regex(phrase_clean)
        matches = len(pattern.findall(text_lower))
        if matches > 0:
            words_in_phrase = len(phrase_clean.split())
            matched_token_count += matches * words_in_phrase

    # Normalize against document length (with a scaling multiplier)
    raw_density = matched_token_count / max(total_tokens, 10)
    score = min(1.0, raw_density * 3.0)
    return round(score, 4)


def extract_tfidf_top_terms(
    text: str,
    reference_corpus: Optional[List[str]] = None,
    top_n: int = 10,
) -> List[str]:
    """
    Extract the top TF-IDF weighted terms for a single document evaluated
    against a reference corpus of legitimate and phishing page texts.
    """
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    if len(tokens) < 2:
        return tokens

    ref = reference_corpus if reference_corpus is not None else REFERENCE_CORPUS
    corpus = ref + [text]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
            max_features=1000,
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Target document is the last row in the matrix
        doc_vector = tfidf_matrix[-1].toarray().flatten()
        feature_names = vectorizer.get_feature_names_out()

        nonzero_indices = doc_vector.nonzero()[0]
        scored_terms = [(feature_names[i], doc_vector[i]) for i in nonzero_indices]
        scored_terms.sort(key=lambda x: x[1], reverse=True)

        return [term for term, _ in scored_terms[:top_n]]
    except Exception:
        # Fallback to simple unique words if vectorizer cannot fit
        return list(dict.fromkeys(tokens))[:top_n]


def extract_content_features(page_text: str) -> Dict[str, Any]:
    """
    Extract NLP-based semantic features from webpage or email text.

    Extracted Features (Section 3.2.1):
    1. urgency_score : float (0.0 - 1.0)
       Density of time-pressuring/urgency phrases (e.g. "immediately", "24 hours", "suspended").
    2. credential_request_score : float (0.0 - 1.0)
       Density of language demanding passwords, PINs, account verification, or financial info.
    3. login_prompt_density : float (0.0 - 1.0)
       Proportion of text containing authentication / login terms.
    4. tfidf_top_terms : List[str]
       Top 10 TF-IDF weighted terms in the document against the reference corpus.
    5. overall_content_risk_score : float (0.0 - 1.0)
       Weighted combination of the three semantic risk scores.

    Parameters
    ----------
    page_text : str
        Raw textual content of the webpage or email.

    Returns
    -------
    dict
        Dictionary of extracted NLP content features.
    """
    # 1. Edge Case Validation
    if not isinstance(page_text, str):
        page_text = ""

    cleaned_text = page_text.strip()
    tokens = _tokenize(cleaned_text)

    # Empty, whitespace-only, or very short text (< 3 words) returns zeroed-out scores
    if len(tokens) < 3:
        return {
            "urgency_score": 0.0,
            "credential_request_score": 0.0,
            "login_prompt_density": 0.0,
            "tfidf_top_terms": tokens[:10],
            "overall_content_risk_score": 0.0,
        }

    # 2. Compute Individual Density Scores
    urgency_score = calculate_phrase_density(cleaned_text, URGENCY_PHRASES, tokens)
    credential_request_score = calculate_phrase_density(cleaned_text, CREDENTIAL_PHRASES, tokens)
    login_prompt_density = calculate_phrase_density(cleaned_text, LOGIN_PROMPT_TERMS, tokens)

    # 3. Extract Top TF-IDF Weighted Terms
    tfidf_top_terms = extract_tfidf_top_terms(cleaned_text, REFERENCE_CORPUS, top_n=10)

    # 4. Overall Content Risk Score Calculation
    # Weights Rationale:
    # - urgency_score (0.45): Urgency and threats of immediate loss/suspension are the primary
    #   manipulation tactic in social engineering and phishing attacks.
    # - credential_request_score (0.35): Direct credential harvesting requests indicate high risk
    #   when combined with urgency.
    # - login_prompt_density (0.20): Form keywords appear on legitimate login pages too, so this
    #   is given a lower weighting to prevent false positives on normal authentication flows.
    raw_overall_risk = (
        (0.45 * urgency_score)
        + (0.35 * credential_request_score)
        + (0.20 * login_prompt_density)
    )
    overall_content_risk_score = round(min(1.0, max(0.0, raw_overall_risk)), 4)

    return {
        "urgency_score": urgency_score,
        "credential_request_score": credential_request_score,
        "login_prompt_density": login_prompt_density,
        "tfidf_top_terms": tfidf_top_terms,
        "overall_content_risk_score": overall_content_risk_score,
    }


def extract_content_features_batch(texts: List[str]) -> pd.DataFrame:
    """
    Extract NLP content features for a batch of page texts and return a pandas DataFrame.

    Parameters
    ----------
    texts : List[str]
        List of text strings to analyze.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per text and columns for each extracted feature.

    Raises
    ------
    TypeError
        If texts is not a list or tuple.
    """
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"texts must be a list or tuple of strings, got {type(texts).__name__}.")

    records = []
    for t in texts:
        features = extract_content_features(t)
        # Store a preview of the text
        preview = (t[:75] + "...") if len(t) > 75 else t
        record = {"text_preview": preview, **features}
        records.append(record)

    return pd.DataFrame(records)
