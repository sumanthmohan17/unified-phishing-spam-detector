"""
Email Text Preprocessing Module
===============================
Module 2: Text normalization and cleaning for email spam/phishing/malware
classification as described in Section 3.2.2 of the project report.
"""

from __future__ import annotations

import html
import re
from typing import List, Set

# Standard English stop words for email text filtering
ENGLISH_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def preprocess_email_text(raw_text: str) -> str:
    """
    Clean, tokenize, and normalize raw email text (plain text or HTML).

    Steps:
    1. Handle empty / None inputs safely.
    2. Unescape HTML entities (e.g., &amp; -> &, &quot; -> ").
    3. Strip HTML markup (<tags>, <a>, <div>, <script>, etc.).
    4. Convert to lowercase.
    5. Remove punctuation and non-alphanumeric noise.
    6. Filter out high-frequency stop words.
    7. Normalize multi-character whitespace to single space.

    Parameters
    ----------
    raw_text : str
        The raw email body text to clean.

    Returns
    -------
    str
        Preprocessed and normalized token string.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # 1. Unescape HTML entities
    text = html.unescape(raw_text)

    # 2. Remove script and style tags with their contents
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Lowercase
    text = text.lower()

    # 5. Extract alphanumeric tokens
    tokens: List[str] = re.findall(r"\b[a-z0-9]{2,}\b", text)

    # 6. Stop words removal
    filtered_tokens: List[str] = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

    # 7. Whitespace normalization
    cleaned_string = " ".join(filtered_tokens).strip()
    return cleaned_string
