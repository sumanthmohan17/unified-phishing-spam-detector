"""
LLM Explainability Layer
========================
Module 3: Natural Language Threat Explanation & Action Recommendation using
the Groq API (llama-3.3-70b-versatile) as described in Section 3.2.3 of the report.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Default active Groq model (note: openai/gpt-oss-120b is available and ultra-fast on Groq)
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are an expert cybersecurity explainability engine in a unified phishing and email spam detection system.
Your role is to explain an ALREADY-COMPUTED detection verdict using the provided detection signals and feature evidence.
Rules:
1. Do NOT re-evaluate or overturn the verdict. Explain why the system reached this conclusion using the supplied features.
2. Do NOT output disclaimers (e.g. 'I am an AI' or 'I am not a security expert').
3. Keep the explanation concise, professional, and clear (2-4 sentences in plain English).
4. Recommend one of three actions:
   - "block": For malicious phishing or malware threats with high-risk signals (e.g., raw IP URLs, redirect chains, brand impersonation, credential theft).
   - "review": For unsolicited commercial spam, borderline marketing, or emails requiring manual confirmation.
   - "safe": For authentic, legitimate communications with no threat signals.
5. Return your output strictly as a JSON object with this format:
{
    "explanation": "<2-4 sentence plain-English explanation>",
    "recommended_action": "block" | "review" | "safe"
}"""


def _get_groq_client() -> Groq:
    """
    Retrieve and validate the Groq API client from environment variables.

    Raises
    ------
    ValueError
        If GROQ_API_KEY is not configured in .env or the system environment.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or "your_groq_api_key" in api_key:
        raise ValueError(
            "GROQ_API_KEY is not set or contains a placeholder. "
            "Please add a valid GROQ_API_KEY to your .env file or environment variables."
        )
    return Groq(api_key=api_key.strip())


def _parse_llm_response(raw_text: str, default_action: str) -> Dict[str, str]:
    """Extract explanation and recommended_action from raw LLM output."""
    try:
        # Try finding JSON block enclosed in markdown or raw JSON
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            explanation = data.get("explanation", "").strip()
            action = data.get("recommended_action", default_action).lower().strip()
            if action not in {"block", "review", "safe"}:
                action = default_action
            if explanation:
                return {"explanation": explanation, "recommended_action": action}
    except Exception:
        pass

    # Fallback to plain text cleaning if JSON parsing fails
    cleaned = re.sub(r"```[a-z]*", "", raw_text).replace("```", "").strip()
    return {"explanation": cleaned, "recommended_action": default_action}


def explain_phishing_detection(
    detection_result: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """
    Generate a plain-English explanation and actionable recommendation for a
    Module 1 Phishing Detection result.

    Parameters
    ----------
    detection_result : dict
        Output dictionary from `phishing_module.classifier.predict()`.
    model_name : str, default="llama-3.3-70b-versatile"
        Groq LLM model name.

    Returns
    -------
    dict
        {
            "explanation": str,
            "recommended_action": "block" | "review" | "safe",
            "raw_llm_response": str,
        }
    """
    verdict = detection_result.get("verdict", "safe").lower()
    confidence = detection_result.get("confidence", 0.0)
    top_features = detection_result.get("top_features", [])
    raw_scores = detection_result.get("raw_scores", {})

    fallback_action = "block" if verdict == "phishing" else "safe"

    user_prompt = f"""Explain this Phishing Detection Verdict:
- Classification Verdict: {verdict.upper()}
- Model Confidence: {confidence:.2%}
- Top Contributing Features: {top_features}

Detailed Multi-Signal Evidence:
• URL Structural Features: {raw_scores.get('url_features', {})}
• Visual Brand Similarity: {raw_scores.get('visual_features', {})}
• NLP Content Analysis: {raw_scores.get('content_features', {})}

Generate a 2-4 sentence plain-English explanation tailored to these exact signals and select the appropriate recommended action."""

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )

        raw_response = response.choices[0].message.content or ""
        parsed = _parse_llm_response(raw_response, default_action=fallback_action)

        return {
            "explanation": parsed["explanation"],
            "recommended_action": parsed["recommended_action"],
            "raw_llm_response": raw_response,
        }
    except Exception as exc:
        print(f"[LLM_LAYER_ERROR] explain_phishing_detection failed: {exc}", file=sys.stderr)
        return {
            "explanation": f"Unable to generate explanation at this time ({type(exc).__name__}: {str(exc)})",
            "recommended_action": fallback_action,
            "raw_llm_response": "",
        }


def explain_spam_detection(
    detection_result: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """
    Generate a plain-English explanation and actionable recommendation for a
    Module 2 Email Spam Classification result with embedded URL risk signals.

    Parameters
    ----------
    detection_result : dict
        Output dictionary from `spam_module.classifier.classify_email()`.
    model_name : str, default="llama-3.3-70b-versatile"
        Groq LLM model name.

    Returns
    -------
    dict
        {
            "explanation": str,
            "recommended_action": "block" | "review" | "safe",
            "raw_llm_response": str,
        }
    """
    verdict = detection_result.get("verdict", "legitimate").lower()
    confidence = detection_result.get("confidence", 0.0)
    embedded_urls = detection_result.get("embedded_urls", [])
    embedded_url_risk = detection_result.get("embedded_url_risk", {})
    top_features = detection_result.get("top_features", [])

    if verdict in {"phishing", "malware"}:
        fallback_action = "block"
    elif verdict == "spam":
        fallback_action = "review"
    else:
        fallback_action = "safe"

    user_prompt = f"""Explain this Email Spam & Threat Classification Verdict:
- Classification Verdict: {verdict.upper()}
- Model Confidence: {confidence:.2%}
- Top Contributing Features: {top_features}
- Embedded URLs Found: {embedded_urls}
- Embedded URL Threat Analysis from Module 1: {embedded_url_risk}

Generate a 2-4 sentence plain-English explanation explaining the verdict (specifically referencing any embedded URL threats if present) and select the appropriate recommended action."""

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )

        raw_response = response.choices[0].message.content or ""
        parsed = _parse_llm_response(raw_response, default_action=fallback_action)

        return {
            "explanation": parsed["explanation"],
            "recommended_action": parsed["recommended_action"],
            "raw_llm_response": raw_response,
        }
    except Exception as exc:
        print(f"[LLM_LAYER_ERROR] explain_spam_detection failed: {exc}", file=sys.stderr)
        return {
            "explanation": f"Unable to generate explanation at this time ({type(exc).__name__}: {str(exc)})",
            "recommended_action": fallback_action,
            "raw_llm_response": "",
        }
