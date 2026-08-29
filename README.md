# Unified Phishing and Email Spam Detection System Using Machine Learning

> Main Project Phase-I (23CS605) — Department of Computer Science and Engineering, Malnad College of Engineering, Hassan (VTU) — 2025-2026

Existing phishing and spam detection systems work in isolation, lacking transparency and real-time adaptability. This project unifies both into a single, explainable pipeline: a multi-signal phishing detector, an email spam classifier, an LLM-generated explanation layer, and real-time cross-validation against live threat intelligence feeds.

**Status: Phase-I in progress.** Modules 1, 2, and 3 are complete and verified. See [Build Status](#build-status) below.

---

## Team

| Name | USN |
|---|---|
| Ritish Sharma | 4MC23CS137 |
| Sumanth Mohan | 4MC23CS162 |
| Tejaswi B N | 4MC23CS172 |
| Jeevan Y R | 4MC24CS408 |

**Guide:** Mr. Keerthi K.S., Assistant Professor, Dept. of CSE, MCE

---

## Overview

The system combines:

- **Phishing Detection Module** — multi-signal ensemble (URL structure, visual brand similarity, NLP page content) using Random Forest + XGBoost
- **Email Spam Classification Module** — Naive Bayes + XGBoost ensemble over TF-IDF features, with embedded URLs automatically routed through the phishing module
- **LLM Explainability Layer** — Groq API-generated plain-English explanations for every detection, with a block/review/safe action recommendation
- **Real-Time Threat Intelligence** — cross-validation against VirusTotal and Google Safe Browsing APIs
- **Deployment** — a Chrome browser extension for real-time in-browser detection, and a Streamlit dashboard for email monitoring and analytics

Full architecture and methodology are documented in the [project report](#report--publication).

---

## Build Status

| Component | Status |
|---|---|
| Module 1 — Phishing Detection: URL structural features | ✅ Complete |
| Module 1 — Phishing Detection: Visual brand similarity | ✅ Complete |
| Module 1 — Phishing Detection: NLP page content analysis | ✅ Complete |
| Module 1 — Phishing Detection: RF + XGBoost ensemble classifier | ✅ Complete |
| Module 2 — Email Spam Classification (TF-IDF + NB + XGBoost) | ✅ Complete |
| Module 2 — Embedded URL routing into Module 1 | ✅ Complete |
| Module 3 — LLM Explainability Layer (Groq API) | ✅ Complete |
| Threat Intelligence Cross-Validation (VirusTotal / Safe Browsing) | 🔲 Not started |
| Chrome Extension | 🔲 Not started |
| Streamlit Dashboard | 🔲 Not started |

**Important:** all testing so far uses synthetic/placeholder training and reference data (hand-constructed URLs, generated logo images, hand-written text samples) to verify code correctness end-to-end. Real-world training and accuracy evaluation on benchmark datasets (PhishTank, UCI ML Phishing Dataset, Enron/SpamAssassin) is planned for a later phase — see [Future Scope](#future-scope). The LLM explanation layer, however, makes real calls to the Groq API and has been verified against actual model output.

---

## Project Structure

```
unified-phishing-spam-detector/
├── requirements.txt
├── .env.example                     # template for required API keys
├── verify_extractor.py              # verification: URL feature extraction
├── verify_visual_similarity.py      # verification: visual brand similarity
├── verify_content_features.py       # verification: NLP content analysis
├── verify_classifier.py             # verification: fused phishing ensemble
├── verify_spam_module.py            # verification: spam classifier + URL routing
├── verify_llm_layer.py              # verification: LLM explainability layer
├── phishing_module/
│   ├── __init__.py
│   ├── feature_extraction.py        # URL structural features
│   ├── visual_similarity.py         # perceptual hashing / brand impersonation detection
│   ├── content_features.py          # TF-IDF + urgency/credential-request scoring
│   ├── classifier.py                # RF + XGBoost stacking ensemble
│   └── test_assets/                 # synthetic reference logos & test images
├── spam_module/
│   ├── __init__.py
│   ├── preprocessing.py             # email text cleaning/normalization
│   ├── url_extraction.py            # embedded URL extraction from email bodies
│   └── classifier.py                # NB + XGBoost ensemble, routes URLs into phishing_module
└── llm_layer/
    ├── __init__.py
    └── explainer.py                 # Groq API plain-English explanations + action recommendation
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip
- A free [Groq API key](https://console.groq.com/keys) (for the explainability layer)

### Installation

```bash
git clone https://github.com/sumanthmohan17/unified-phishing-spam-detector.git
cd unified-phishing-spam-detector
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your own Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

Never commit `.env` — it's already excluded via `.gitignore`.

### Running the verification scripts

Each module component has a standalone verification script that exercises it against sample/synthetic data:

```bash
python verify_extractor.py
python verify_visual_similarity.py
python verify_content_features.py
python verify_classifier.py
python verify_spam_module.py
python verify_llm_layer.py    # requires GROQ_API_KEY to be set
```

---

## Tech Stack

- **ML/Data:** scikit-learn, XGBoost, pandas, NumPy
- **NLP:** TF-IDF (scikit-learn), custom phrase-density scoring
- **Visual analysis:** Pillow, imagehash (perceptual hashing)
- **URL parsing:** tldextract, python-whois
- **LLM:** Groq API (openai/gpt-oss-120b), python-dotenv
- **Planned:** VirusTotal & Google Safe Browsing APIs (threat intel), Streamlit (dashboard), Chrome Extension APIs

---

## Future Scope

- Train the ensemble on real benchmark datasets (PhishTank, UCI ML Phishing Dataset, Enron/SpamAssassin) and evaluate against literature baselines
- Integrate live VirusTotal / Google Safe Browsing cross-validation
- Build and deploy the Chrome extension for real-time in-browser detection
- Develop the Streamlit dashboard for email scanning and threat visualization
- Extend the visual brand similarity module with a larger, real reference library of brand identities
- Evaluate LLM explanation quality with real users

---

## Report & Publication

- Full Phase-I project report: *"Unified Phishing and Email Spam Detection System Using Machine Learning"*
- Published survey paper: *"A Survey on Unified Phishing and Email Spam Detection Systems Using Machine Learning and NLP"* — International Journal of Scientific Research in Engineering and Management (IJSREM), Volume 10, Issue 5, May 2026. DOI: [10.55041/IJSREM63330](https://doi.org/10.55041/IJSREM63330)

---

## License

Academic project — Malnad College of Engineering, Hassan. License to be added.
