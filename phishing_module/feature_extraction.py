"""
URL Structural Feature Extractor
================================
Module 1: URL Feature Extraction component for the Unified Phishing and
Email Spam Detection System.

Extracts lexical, structural, and host-based features from raw URL strings
as described in Section VI-A ("URL Structural Features") of the project report.
"""

from __future__ import annotations

import datetime
import ipaddress
import math
import re
import urllib.parse
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
import tldextract
import whois

# Known URL shortener services commonly leveraged in redirect chains / evasion
KNOWN_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "lnkd.in",
    "rebrand.ly",
    "tiny.cc",
    "qr.ae",
    "ift.tt",
    "curt.ly",
    "s.id",
    "cutt.ly",
    "v.gd",
    "trib.al",
    "clck.ru",
    "shorturl.at",
    "tiny.one",
    "soo.gd",
    "bc.vc",
}


def validate_url(url: str) -> str:
    """
    Validate that the provided input is a well-formed URL string.

    Parameters
    ----------
    url : str
        The URL string to validate.

    Returns
    -------
    str
        The cleaned URL string.

    Raises
    ------
    TypeError
        If the input is not a string.
    ValueError
        If the URL string is empty or lacks a valid scheme/network location.
    """
    if not isinstance(url, str):
        raise TypeError(f"URL must be a string, got {type(url).__name__}.")

    cleaned_url = url.strip()
    if not cleaned_url:
        raise ValueError("URL cannot be empty or solely whitespace.")

    try:
        parsed = urllib.parse.urlparse(cleaned_url)
    except Exception as exc:
        raise ValueError(f"Failed to parse URL '{cleaned_url}': {exc}") from exc

    if not parsed.scheme:
        raise ValueError(
            f"Invalid URL '{cleaned_url}': Missing URL scheme (e.g., 'http://' or 'https://')."
        )

    if parsed.scheme.lower() not in {"http", "https", "ftp"}:
        raise ValueError(
            f"Invalid URL '{cleaned_url}': Unsupported scheme '{parsed.scheme}'. Expected 'http' or 'https'."
        )

    if not parsed.netloc and not parsed.hostname:
        raise ValueError(
            f"Invalid URL '{cleaned_url}': Missing network location / host address."
        )

    return cleaned_url


def calculate_url_entropy(url_str: str) -> float:
    """
    Calculate the Shannon entropy of a URL string.

    Shannon entropy measures the uncertainty / randomness of characters in the URL:
        H(X) = - sum(p(x) * log2(p(x)))

    High entropy often indicates obfuscation, randomly generated tokens, or algorithmic domain generation.

    Parameters
    ----------
    url_str : str
        Input string.

    Returns
    -------
    float
        Shannon entropy in bits, rounded to 4 decimal places.
    """
    if not url_str:
        return 0.0

    length = len(url_str)
    counts = Counter(url_str)
    entropy = -sum((count / length) * math.log2(count / length) for count in counts.values())
    return round(entropy, 4)


def is_ip_address(host: str) -> bool:
    """
    Check if the host string is a raw IPv4 or IPv6 address.

    Parameters
    ----------
    host : str
        Hostname or IP string.

    Returns
    -------
    bool
        True if the host is a valid IPv4 or IPv6 address, False otherwise.
    """
    if not host:
        return False

    clean_host = host.strip("[]").split(":")[0]  # Remove IPv6 brackets and port if present
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def get_domain_age_days(domain_name: str) -> Optional[int]:
    """
    Query the domain registration age in days using WHOIS.

    If the domain lookup fails, times out, is an IP address, or does not
    contain creation date metadata, returns None without raising an exception.

    Parameters
    ----------
    domain_name : str
        The registered domain name to query (e.g. 'google.com').

    Returns
    -------
    Optional[int]
        Age in days since domain registration, or None if unavailable.
    """
    if not domain_name or is_ip_address(domain_name):
        return None

    try:
        w = whois.whois(domain_name)
        creation_date = w.creation_date

        if not creation_date:
            return None

        # Handle cases where multiple creation dates are returned (list)
        if isinstance(creation_date, list):
            valid_dates = [d for d in creation_date if isinstance(d, (datetime.datetime, datetime.date))]
            if not valid_dates:
                return None
            creation_date = min(valid_dates)

        # Ensure datetime object
        if isinstance(creation_date, datetime.date) and not isinstance(creation_date, datetime.datetime):
            creation_date = datetime.datetime.combine(creation_date, datetime.time.min)

        if not isinstance(creation_date, datetime.datetime):
            return None

        # Calculate difference with timezone handling
        if creation_date.tzinfo is not None:
            now = datetime.datetime.now(creation_date.tzinfo)
        else:
            now = datetime.datetime.now()

        age = (now - creation_date).days
        return max(0, age) if age >= 0 else None

    except Exception:
        # Gracefully handle WHOIS lookup failures, socket timeouts, rate limits, etc.
        return None


def extract_url_features(url: str) -> Dict[str, Any]:
    """
    Extract 9 structural and host-based features from a URL string.

    Extracted Features (Section VI-A):
    1. url_length : int
       Total character count of the URL.
    2. subdomain_count : int
       Number of subdomain levels (0 for root domain, e.g. www=1, a.b=2).
    3. has_ip_address : bool
       Whether the hostname is a raw IPv4 or IPv6 address.
    4. uses_https : bool
       Whether the URL uses the HTTPS protocol.
    5. hyphen_count : int
       Count of hyphens in the domain name.
    6. special_char_count : int
       Count of non-alphanumeric special characters in the full URL.
    7. url_entropy : float
       Shannon entropy of the URL string.
    8. domain_age_days : Optional[int]
       Registration age of domain in days (via WHOIS; None if failed/IP).
    9. has_redirect_chain : bool
       Whether the URL or its query/redirect parameter uses a known shortener domain.

    Parameters
    ----------
    url : str
        URL string to analyze.

    Returns
    -------
    dict
        Dictionary containing the 9 extracted features.

    Raises
    ------
    TypeError
        If url is not a string.
    ValueError
        If url is empty or malformed.
    """
    cleaned_url = validate_url(url)
    parsed = urllib.parse.urlparse(cleaned_url)
    extracted = tldextract.extract(cleaned_url)

    # 1. url_length
    url_length = len(cleaned_url)

    # 2. subdomain_count
    subdomain_str = extracted.subdomain.strip()
    if subdomain_str:
        subdomain_parts = [part for part in subdomain_str.split(".") if part]
        subdomain_count = len(subdomain_parts)
    else:
        subdomain_count = 0

    # 3. has_ip_address
    hostname = parsed.hostname or extracted.domain or ""
    has_ip = is_ip_address(hostname)
    # If tldextract indicates IP address or hostname is an IP
    if not has_ip and extracted.ipv4:
        has_ip = True

    # 4. uses_https
    uses_https = (parsed.scheme.lower() == "https")

    # 5. hyphen_count (in domain name)
    domain_part = extracted.domain if extracted.domain else hostname
    hyphen_count = domain_part.count("-")

    # 6. special_char_count (non-alphanumeric characters in full URL)
    special_char_count = sum(1 for c in cleaned_url if not c.isalnum())

    # 7. url_entropy
    url_entropy = calculate_url_entropy(cleaned_url)

    # 8. domain_age_days
    if has_ip:
        domain_age_days = None
    else:
        # Use registered domain (e.g., 'google.com' or 'example.co.uk')
        target_domain = extracted.registered_domain or hostname
        domain_age_days = get_domain_age_days(target_domain)

    # 9. has_redirect_chain: check if main host is a shortener OR if any query parameter targets a shortener
    registered_domain = extracted.registered_domain.lower() if extracted.registered_domain else ""
    host_lower = hostname.lower()

    is_shortener_domain = (
        host_lower in KNOWN_SHORTENERS
        or registered_domain in KNOWN_SHORTENERS
    )

    has_query_shortener = False
    if parsed.query:
        query_dict = urllib.parse.parse_qs(parsed.query)
        for values in query_dict.values():
            for v in values:
                v_clean = v.strip()
                if not v_clean:
                    continue
                # Normalize scheme to parse domain/host if missing
                test_v = v_clean if v_clean.startswith(("http://", "https://", "//")) else f"http://{v_clean}"
                try:
                    p_v = urllib.parse.urlparse(test_v)
                    v_host = (p_v.hostname or "").lower()
                    if v_host in KNOWN_SHORTENERS:
                        has_query_shortener = True
                        break
                    v_ext = tldextract.extract(test_v)
                    if (v_ext.registered_domain or "").lower() in KNOWN_SHORTENERS:
                        has_query_shortener = True
                        break
                except Exception:
                    pass
            if has_query_shortener:
                break

    has_redirect_chain = is_shortener_domain or has_query_shortener

    return {
        "url_length": url_length,
        "subdomain_count": subdomain_count,
        "has_ip_address": has_ip,
        "uses_https": uses_https,
        "hyphen_count": hyphen_count,
        "special_char_count": special_char_count,
        "url_entropy": url_entropy,
        "domain_age_days": domain_age_days,
        "has_redirect_chain": has_redirect_chain,
    }


def extract_url_features_batch(urls: List[str]) -> pd.DataFrame:
    """
    Extract structural features for a batch of URLs and return a pandas DataFrame.

    Parameters
    ----------
    urls : List[str]
        List of URL strings to analyze.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per URL and one column per feature, including
        an initial 'url' column.

    Raises
    ------
    TypeError
        If urls is not a list/iterable or any URL is not a string.
    ValueError
        If any URL in the batch is malformed.
    """
    if not isinstance(urls, (list, tuple)):
        raise TypeError(f"urls must be a list or tuple of strings, got {type(urls).__name__}.")

    records = []
    for raw_url in urls:
        features = extract_url_features(raw_url)
        record = {"url": raw_url, **features}
        records.append(record)

    df = pd.DataFrame(records)
    return df
