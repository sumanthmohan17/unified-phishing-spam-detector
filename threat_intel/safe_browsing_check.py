"""
Google Safe Browsing Threat Intelligence Integration Module
============================================================
Module 4: Real-time URL threat cross-validation using Google Safe Browsing API v4
as described in Section 3.2.1 and Section VIII-D of the project report.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def _get_safe_browsing_api_key() -> str:
    """Retrieve and validate the Google Safe Browsing API key from environment."""
    api_key = os.environ.get("SAFE_BROWSING_API_KEY")
    if not api_key or api_key.strip() == "" or "your_safe_browsing_api_key" in api_key:
        raise ValueError(
            "SAFE_BROWSING_API_KEY is not configured in .env or environment variables. "
            "Please set a valid SAFE_BROWSING_API_KEY."
        )
    return api_key.strip()


def check_url_safe_browsing(
    url: str,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Query Google Safe Browsing API v4 threatMatches:find endpoint.

    Parameters
    ----------
    url : str
        The URL to inspect.
    timeout : int, default=15
        HTTP request timeout in seconds.

    Returns
    -------
    dict
        {
            "flagged": bool,
            "threat_types": List[str],
            "raw_response": dict,
            "error": Optional[str],
        }
    """
    if not url or not isinstance(url, str):
        return {
            "flagged": False,
            "threat_types": [],
            "raw_response": {},
            "error": "Invalid or empty URL provided",
        }

    try:
        api_key = _get_safe_browsing_api_key()
    except ValueError as exc:
        return {
            "flagged": False,
            "threat_types": [],
            "raw_response": {},
            "error": str(exc),
        }

    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"

    payload = {
        "client": {
            "clientId": "unified-phishing-detector",
            "clientVersion": "1.0.0",
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)

        if resp.status_code == 200:
            data = resp.json()
            matches = data.get("matches", [])
            threat_types = [m.get("threatType") for m in matches if m.get("threatType")]
            flagged = len(matches) > 0

            return {
                "flagged": flagged,
                "threat_types": threat_types,
                "raw_response": data,
                "error": None,
            }
        else:
            return {
                "flagged": False,
                "threat_types": [],
                "raw_response": resp.json() if resp.content else {},
                "error": f"Safe Browsing API HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as exc:
        return {
            "flagged": False,
            "threat_types": [],
            "raw_response": {},
            "error": f"Safe Browsing connection error: {type(exc).__name__}: {str(exc)}",
        }
