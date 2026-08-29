"""
Email URL Extraction Module
===========================
Module 2: Robust regex-based extraction of embedded hyperlinks and raw URLs from
email bodies (plain text and HTML) as described in Section VIII-C of the report.
"""

from __future__ import annotations

import html
import re
from typing import List, Set


def extract_urls_from_email(email_body: str) -> List[str]:
    """
    Extract all distinct URLs found in an email body (plain text or HTML).

    Extracts:
    - HTML anchor href attributes: <a href="https://example.com">
    - HTML src attributes: <img src="http://example.com/img.png">
    - Plain text HTTP / HTTPS URLs: https://www.google.com
    - Raw IP address URLs: http://192.168.1.1/login.php
    - Web / shortener URL patterns: http://bit.ly/xyz

    Parameters
    ----------
    email_body : str
        The raw email body text or HTML content.

    Returns
    -------
    list[str]
        Deduplicated list of extracted URL strings in order of occurrence.
    """
    if not email_body or not isinstance(email_body, str):
        return []

    # Unescape HTML entities so &amp; becomes & inside URLs
    content = html.unescape(email_body)

    extracted_urls: List[str] = []
    seen: Set[str] = set()

    def _add_url(u: str):
        # Strip enclosing quotes, whitespace, and trailing punctuation like periods or parentheses
        cleaned = u.strip("\"'<>[](){} \t\n\r")
        # Remove trailing sentence punctuation
        cleaned = re.sub(r"[.,;!?]+$", "", cleaned)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            extracted_urls.append(cleaned)

    # 1. Extract from HTML href and src attributes
    html_attr_pattern = r"""(?:href|src)\s*=\s*["']([^"'>\s]+)["']"""
    for match in re.findall(html_attr_pattern, content, flags=re.IGNORECASE):
        if re.match(r"^(?:https?|ftp)://", match, flags=re.IGNORECASE):
            _add_url(match)

    # 2. Extract plain text URLs with scheme (http://, https://, ftp://)
    scheme_url_pattern = r"""(?i)\b((?:https?|ftp)://[^\s<>"'{}|\\^`]+)"""
    for match in re.findall(scheme_url_pattern, content):
        _add_url(match)

    # 3. Extract plain www URLs (e.g. www.domain.com/path)
    www_pattern = r"""(?i)\b(www\.[a-zA-Z0-9.\-_]+\.[a-zA-Z]{2,}[^\s<>"'{}|\\^`]*)\b"""
    for match in re.findall(www_pattern, content):
        formatted = f"http://{match}"
        _add_url(formatted)

    return extracted_urls
