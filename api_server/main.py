"""
Unified Phishing Detection Local API Server
===========================================
FastAPI backend service exposing Module 1 (Phishing Classifier),
Module 4 (Threat Intelligence Cross-Validation), and Module 3 (LLM Explainability)
for Chrome Extension and client integration as described in Section 3.2.4.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
load_dotenv()

# Import detection modules
import phishing_module
from phishing_module.classifier import generate_synthetic_training_data, predict, train_ensemble
from phishing_module.visual_similarity import build_reference_library

import threat_intel
from threat_intel.cross_validator import enrich_detection_with_threat_intel

import llm_layer
from llm_layer.explainer import explain_phishing_detection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_server")

# In-memory cached model and reference library
GLOBAL_STATE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cache the ensemble model in memory at server startup."""
    logger.info("Initializing Phishing Detection Stacking Ensemble model...")
    X_syn, y_syn = generate_synthetic_training_data(n_samples=100, random_state=42)
    model, metrics = train_ensemble(X_syn, y_syn, random_state=42)
    GLOBAL_STATE["phishing_model"] = model
    GLOBAL_STATE["model_metrics"] = metrics
    logger.info(f"Model initialized successfully. Accuracy on validation split: {metrics['accuracy']:.2%}")
    yield
    GLOBAL_STATE.clear()


app = FastAPI(
    title="Unified Phishing Detection API",
    description="Local REST API backend for real-time URL phishing detection, threat intel cross-validation, and LLM explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS strictly for Chrome Extension and localhost/127.0.0.1 origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"^(chrome-extension://[a-zA-Z0-9]+|http://localhost(?::\d+)?|http://127\.0\.0\.1(?::\d+)?)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeURLRequest(BaseModel):
    url: str = Field(..., description="The website URL to inspect for phishing threats", min_length=1)
    check_threat_intel: bool = Field(True, description="Whether to cross-validate against VirusTotal and Google Safe Browsing")
    generate_explanation: bool = Field(True, description="Whether to generate a natural language explanation via Groq LLM")


class LLMExplanation(BaseModel):
    explanation: str
    recommended_action: str


class AnalyzeURLResponse(BaseModel):
    url: str
    verdict: str
    confidence: float
    phishing_probability: float
    verdict_source: str
    top_features: List[str]
    raw_scores: Dict[str, Any]
    threat_intel: Optional[Dict[str, Any]] = None
    llm_explanation: Optional[LLMExplanation] = None


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler returning clean JSON error responses."""
    logger.error(f"Unhandled server error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "type": type(exc).__name__,
        },
    )


@app.get("/health", summary="Health Check")
async def health_check():
    """Simple health check endpoint for Chrome extension connectivity verification."""
    return {
        "status": "ok",
        "service": "phishing-detector-api",
        "model_loaded": "phishing_model" in GLOBAL_STATE,
    }


@app.post("/analyze-url", response_model=AnalyzeURLResponse, summary="Analyze URL for Phishing Threats")
async def analyze_url(req: AnalyzeURLRequest):
    """
    Analyze a URL for phishing threats using ML, Threat Intel, and LLM Explainability.

    Execution Pipeline:
    1. Runs URL structural feature extraction and Stacking Ensemble prediction (URL-only signals).
    2. (Optional) Enriches detection with live VirusTotal and Google Safe Browsing cross-validation.
    3. (Optional) Generates tailored natural language threat explanation and action recommendation via Groq.
    """
    target_url = req.url.strip()
    if not target_url or not (target_url.startswith("http://") or target_url.startswith("https://") or target_url.startswith("ftp://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL: URL must start with a valid scheme ('http://', 'https://', or 'ftp://').",
        )

    model = GLOBAL_STATE.get("phishing_model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Detection model is not initialized.",
        )

    try:
        # 1. Run ML Phishing Prediction (URL-only signals, no screenshot available at extension URL-ping time)
        base_prediction = predict(
            url=target_url,
            screenshot_path=None,
            page_text="",
            model=model,
            reference_dir=None,
            precomputed_library=None,
        )

        # 2. Real-time Threat Intelligence Cross-Validation & Escalation
        if req.check_threat_intel:
            detection_result = enrich_detection_with_threat_intel(
                detection_result=base_prediction,
                url=target_url,
                enable_vt_rate_limit=True,
            )
        else:
            detection_result = dict(base_prediction)
            detection_result["verdict_source"] = "ml_model"
            detection_result["threat_intel"] = None

        # 3. LLM Natural Language Explainability Layer
        llm_explanation_obj: Optional[LLMExplanation] = None
        if req.generate_explanation:
            expl_res = explain_phishing_detection(detection_result)
            llm_explanation_obj = LLMExplanation(
                explanation=expl_res.get("explanation", "No explanation available"),
                recommended_action=expl_res.get("recommended_action", "review"),
            )

        return AnalyzeURLResponse(
            url=target_url,
            verdict=detection_result.get("verdict", "safe"),
            confidence=round(float(detection_result.get("confidence", 0.0)), 4),
            phishing_probability=round(float(detection_result.get("phishing_probability", 0.0)), 4),
            verdict_source=detection_result.get("verdict_source", "ml_model"),
            top_features=detection_result.get("top_features", []),
            raw_scores=detection_result.get("raw_scores", {}),
            threat_intel=detection_result.get("threat_intel"),
            llm_explanation=llm_explanation_obj,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error analyzing URL '{target_url}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing URL: {type(exc).__name__}: {str(exc)}",
        )
