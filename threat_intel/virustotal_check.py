"""
VirusTotal Threat Intelligence Integration Module
==================================================
Module 4: Real-time URL threat cross-validation using VirusTotal API v3
with automated sliding-window rate limiting (4 requests/minute) as described
in Section 3.2.1 and Section VIII-D of the project report.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class VirusTotalRateLimiter:
    """
    Thread-safe sliding-window rate limiter enforcing VirusTotal's
    free-tier limit of 4 requests per 60 seconds (15s average spacing).
    """

    def __init__(self, max_requests: int = 4, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """
        Wait if necessary to ensure rate limit compliance, then record timestamp.
        Returns the duration slept in seconds.
        """
        slept_duration = 0.0
        with self._lock:
            now = time.time()
            # Prune timestamps outside the active window
            self.timestamps = [t for t in self.timestamps if now - t < self.window_seconds]

            if len(self.timestamps) >= self.max_requests:
                oldest = self.timestamps[0]
                sleep_needed = (oldest + self.window_seconds) - now + 0.1
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                    slept_duration = sleep_needed
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < self.window_seconds]

            self.timestamps.append(time.time())
        return slept_duration


# Global rate limiter instance for VirusTotal
_vt_limiter = VirusTotalRateLimiter(max_requests=4, window_seconds=60.0)


def _get_virustotal_api_key() -> str:
    """Retrieve and validate the VirusTotal API key from environment."""
    api_key = os.environ.get("VIRUSTOTAL_API_KEY")
    if not api_key or api_key.strip() == "" or "your_virustotal_api_key" in api_key:
        raise ValueError(
            "VIRUSTOTAL_API_KEY is not configured in .env or environment variables. "
            "Please set a valid VIRUSTOTAL_API_KEY."
        )
    return api_key.strip()


def check_url_virustotal(
    url: str,
    enable_rate_limit: bool = True,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Query VirusTotal API v3 for URL threat analysis report.

    Parameters
    ----------
    url : str
        The URL to inspect.
    enable_rate_limit : bool, default=True
        Whether to enforce the 4 requests/minute rate limit throttle.
    timeout : int, default=15
        HTTP request timeout in seconds.

    Returns
    -------
    dict
        {
            "flagged": bool,
            "malicious_count": int,
            "total_engines": int,
            "raw_response": dict,
            "error": Optional[str],
        }
    """
    if not url or not isinstance(url, str):
        return {
            "flagged": False,
            "malicious_count": 0,
            "total_engines": 0,
            "raw_response": {},
            "error": "Invalid or empty URL provided",
        }

    try:
        api_key = _get_virustotal_api_key()
    except ValueError as exc:
        return {
            "flagged": False,
            "malicious_count": 0,
            "total_engines": 0,
            "raw_response": {},
            "error": str(exc),
        }

    # Enforce VirusTotal 4 req/min rate limit
    if enable_rate_limit:
        _vt_limiter.acquire()

    try:
        # Base64 URL identifier (RFC 4648 URL-safe without padding)
        url_id = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").strip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        headers = {
            "x-apikey": api_key,
            "Accept": "application/json",
        }

        resp = requests.get(endpoint, headers=headers, timeout=timeout)

        if resp.status_code == 200:
            payload = resp.json()
            attributes = payload.get("data", {}).get("attributes", {})
            stats = attributes.get("last_analysis_stats", {})

            malicious_count = int(stats.get("malicious", 0))
            suspicious_count = int(stats.get("suspicious", 0))
            total_engines = sum(int(v) for v in stats.values()) if stats else 0

            # Flagged if any security engine detected malicious threat
            flagged = (malicious_count > 0) or (suspicious_count > 0)

            return {
                "flagged": flagged,
                "malicious_count": malicious_count,
                "total_engines": total_engines,
                "raw_response": payload,
                "error": None,
            }
        elif resp.status_code == 404:
            # URL not found in VirusTotal cache
            return {
                "flagged": False,
                "malicious_count": 0,
                "total_engines": 0,
                "raw_response": resp.json() if resp.content else {},
                "error": "URL not found in VirusTotal analysis database",
            }
        elif resp.status_code == 429:
            return {
                "flagged": False,
                "malicious_count": 0,
                "total_engines": 0,
                "raw_response": resp.json() if resp.content else {},
                "error": "VirusTotal API rate limit exceeded (HTTP 429)",
            }
        else:
            return {
                "flagged": False,
                "malicious_count": 0,
                "total_engines": 0,
                "raw_response": resp.json() if resp.content else {},
                "error": f"VirusTotal API HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as exc:
        return {
            "flagged": False,
            "malicious_count": 0,
            "total_engines": 0,
            "raw_response": {},
            "error": f"VirusTotal connection error: {type(exc).__name__}: {str(exc)}",
        }
