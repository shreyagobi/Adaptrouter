---
title: AdaptRouter
emoji: 🐢
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.35.0"
python_version: '3.10'
app_file: app.py
pinned: false
license: mit
short_description: AdaptRouter — a self-improving routing middleware for LLMs
---

<div align="center">

# 🔀 AdaptRouter

### Self-Improving LLM Routing Middleware

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)

*Routes queries to the right LLM model, explains why, and gets smarter from your feedback — automatically.*

</div>

---

## 📖 Overview

**AdaptRouter** is a self-improving LLM routing middleware that intelligently routes user queries to either a **fast model** (Llama 3.1 8B) or a **smart model** (Llama 3.3 70B) based on query complexity. Unlike static routers such as [RouteLLM](https://github.com/lm-sys/RouteLLM), AdaptRouter incorporates a **feedback loop** that continuously retrains the routing classifier from real usage signals, improving accuracy over time without manual intervention.

### Key Differentiators

| Capability | Static Routers (RouteLLM) | **AdaptRouter** |
|---|---|---|
| Routing accuracy | Fixed after training | ✅ Improves with feedback |
| Domain adaptation | ❌ General only | ✅ Adapts to any domain |
| Explainability | ❌ Black box | ✅ SHAP-based explanations |
| Confidence calibration | ❌ No analysis | ✅ ECE + reliability diagrams |
| Drift detection | ❌ None | ✅ Bootstrap-calibrated alerts |
| Self-retraining | ❌ Manual | ✅ Automatic after 20 signals |
| Implicit feedback | ❌ None | ✅ Rephrasing, escalation, acceptance |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         AdaptRouter.route()     │
│  ┌───────────────────────────┐  │
│  │  1. Input Validation      │  │  ← Long query guard (>150 words → complex)
│  │  2. Embedding (384-dim)   │  │  ← sentence-transformers
│  │  3. Classification (~5ms) │  │  ← LogisticRegression
│  │  4. Confidence Threshold  │  │  ← 0.65 threshold, fallback to smart
│  │  5. Route to Model        │  │  ← Groq API (fast or smart)
│  │  6. Log Decision (SQLite) │  │  ← FeedbackStore
│  │  7. Implicit Feedback     │  │  ← Rephrasing, escalation detection
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    │                         │
    ▼                         ▼
┌──────────┐          ┌──────────────┐
│ Response │          │   Feedback   │
│ to User  │          │   Loop       │
└──────────┘          │  ┌────────┐  │
                      │  │ Store  │──┼──→ SQLite DB
                      │  │Feedback│  │
                      │  └────────┘  │
                      │  ┌────────┐  │
                      │  │Retrain │──┼──→ Warm-start + Platt calibration
                      │  │(auto)  │  │     after 20 feedback signals
                      │  └────────┘  │
                      │  ┌────────┐  │
                      │  │ SHAP   │──┼──→ Word-level routing explanations
                      │  │Explain │  │
                      │  └────────┘  │
                      └──────────────┘
```

### How Routing Works

1. **Embed** the query into a 384-dimensional vector using `sentence-transformers`
2. **Classify** with a `LogisticRegression` classifier in ~5ms locally
3. **Check confidence**: if `P(simple) ≥ 0.65` and label is `simple` → trusted → fast model
4. **Fallback**: low confidence or `complex` label → smart model
5. **Answer**: call the chosen model via Groq API
6. **Log**: store decision + answer in SQLite for future retraining
7. **Learn**: detect implicit feedback signals (rephrasing, escalation, topic change)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Groq API key](https://console.groq.com/) (free tier available)
- The base [llm-router](https://github.com/shreyagobi/adaptrouter) project (provides the base classifier and embedder)

### Installation

```bash
# Clone the repository
git clone https://github.com/shreyagobi/adaptrouter.git
cd adaptrouter

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package with optional extras
pip install -e ".[all]"
```

### Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_ROUTER_PATH=/path/to/your/llm-router
```

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key for LLM inference |
| `LLM_ROUTER_PATH` | Absolute path to the base `llm-router` project directory |
| `ADAPTROUTER_TELEMETRY` | Set to `true` to opt into anonymous telemetry (default: `false`) |

### Run the Dashboard

```bash
# Run the Streamlit dashboard
streamlit run app.py

# Or run from the dashboard directory
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

### Programmatic Usage

```python
from adaptrouter import AdaptRouter

# Initialize the router
router = AdaptRouter(domain="my_app")

# Route a query — returns answer + routing metadata
result = router.route("What is the capital of France?")
print(result["answer"])       # "Paris is the capital of France."
print(result["model_used"])   # "llama-3.1-8b-instant" (fast model)
print(result["confidence"])   # 0.92
print(result["latency_s"])    # 0.46

# Provide feedback — this trains the router
router.feedback(result["query_id"], was_helpful=True)

# After 20 feedback signals → automatic retraining
# The router gets smarter for your specific domain!
```

---

## 📦 Module Reference

### Core Package (`adaptrouter/`)

#### `core.py` — AdaptRouter

The main entry point. Orchestrates the entire routing pipeline.

```python
class AdaptRouter:
    def __init__(self, mode="lr", domain="general")
    def classify(self, query: str) -> dict       # Classification only (no API call)
    def route(self, query: str) -> dict           # Full pipeline: classify → answer → log
    def feedback(self, query_id, was_helpful, rating=None) -> dict
    def get_stats(self) -> dict                   # Performance and feedback statistics
```

**Route response schema:**

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original query text |
| `query_id` | `str` | Unique 12-char ID for feedback linkage |
| `answer` | `str` | Generated answer from the chosen model |
| `label` | `str` | `"simple"` or `"complex"` |
| `confidence` | `float` | Classifier confidence (0.0–1.0) |
| `p_simple` | `float` | Probability of being simple |
| `p_complex` | `float` | Probability of being complex |
| `trusted` | `bool` | Whether confidence exceeds threshold |
| `model_used` | `str` | Full model name used for inference |
| `latency_s` | `float` | End-to-end latency in seconds |
| `total_tokens` | `int` | Total tokens consumed |
| `routing_reason` | `str` | Human-readable routing explanation |

---

#### `feedback_store.py` — FeedbackStore

SQLite-backed persistence layer for all routing decisions, feedback signals, and retraining history.

**Database schema** (3 tables):

- **`routing_decisions`** — Every query classified by the router  
- **`feedback`** — Explicit (thumbs up/down, 1–5 rating) and implicit signals  
- **`retraining_history`** — Log of every retrain event with accuracy before/after  

```python
class FeedbackStore:
    def store_decision(self, result: dict)
    def store_feedback(self, query_id, was_helpful=None, user_rating=None,
                       implicit_type=None, implicit_conf=None)
    def get_labelled_feedback(self, unused_only=True) -> list
    def count_new_labelled(self) -> int
    def mark_feedback_as_used(self, feedback_ids: list)
    def log_retrain_event(self, old_acc, new_acc, n_examples, status, notes="")
    def hours_since_last_retrain(self) -> float
    def get_stats(self) -> dict
```

---

#### `feedback_collector.py` — FeedbackCollector

Facade class that orchestrates explicit feedback, implicit signal detection, and retraining triggers. Implements the **Facade pattern** — developers interact only with this class.

**Retraining conditions (all must be true):**
1. ≥ 20 new labelled examples (`RETRAIN_THRESHOLD`)
2. ≥ 1 hour since last retrain (`MIN_RETRAIN_INTERVAL_HOURS`)
3. A retrainer instance is available

Retraining runs in a **background thread** so it never blocks user-facing requests.

---

#### `implicit_feedback.py` — ImplicitFeedbackDetector

Detects routing quality signals from user behaviour patterns without explicit button clicks. Research shows only 1–5% of users give explicit feedback; implicit signals are 20x more abundant.

| Signal | Detection Method | Confidence | Interpretation |
|---|---|---|---|
| **Rephrasing** | Cosine similarity > 0.82 within 60s | 0.60–0.85 | Previous answer was insufficient |
| **Escalation** | Keyword matching ("explain more", "go deeper", etc.) | 0.80 | Fast model answer was too shallow |
| **Acceptance** | Cosine similarity < 0.35 (topic change) | 0.70 | Previous answer was satisfactory |
| **Abandonment** | Session inactivity analysis | 0.50 | Answer was unsatisfactory |

---

#### `retrainer.py` — AdaptiveRetrainer

Handles warm-start retraining with Platt calibration.

**Retraining pipeline:**

1. Fetch unused labelled feedback from SQLite
2. Combine with original training data (feedback weighted 2x)
3. Embed all texts using sentence-transformers
4. **Warm-start**: copy current classifier weights → fit on combined data
5. **Platt scaling**: wrap with `CalibratedClassifierCV(method="sigmoid")` for reliable probabilities
6. Validate on held-out set (20 queries) — reject if accuracy drops > 2%
7. Save accepted classifier to disk; mark feedback as used
8. Log retrain event with before/after accuracy

---

#### `explainer.py` — RoutingExplainer

SHAP-based explainability for every routing decision.

**How it works:**
- Uses `shap.LinearExplainer` (exact for logistic regression — no approximation)
- Computes SHAP values for each embedding dimension
- Approximates word-level contributions by projecting individual word embeddings onto the SHAP direction vector
- Generates human-readable summaries ("Key words driving this: explain, tradeoffs, architecture")

```python
from adaptrouter.explainer import RoutingExplainer

explainer = RoutingExplainer(classifier=clf, background_size=20)
explanation = explainer.explain("Explain how attention works in transformers")

print(explanation["shap_summary"])
# "Classified as COMPLEX with 87.3% confidence. Key words driving this:
#  attention, transformers, explain. Strongest signal from embedding
#  dimension 142 (SHAP=+0.0341)."

print(explanation["word_scores"])
# [{"word": "transformers", "score": 0.1823, "direction": "→ complex"},
#  {"word": "attention",    "score": 0.1456, "direction": "→ complex"}, ...]
```

---

#### `calibration.py` — CalibrationAnalyzer

Measures confidence calibration using **Expected Calibration Error (ECE)**.

- ECE < 0.05: Excellent — confidence scores are very reliable
- ECE < 0.10: Good — mostly reliable
- ECE < 0.15: Fair — consider Platt scaling
- ECE ≥ 0.15: Poor — scores may be misleading

Generates **reliability diagrams** (calibration curves) with confidence distribution histograms.

---

#### `drift_detector.py` — DriftDetector

Detects when live query distribution shifts away from training data using embedding space centroid distance.

- Computes training data centroid (mean embedding)
- Measures L2 distance between training centroid and live query centroid
- **Bootstrap-calibrated threshold**: 95th percentile of 100 bootstrap samples
- Severity levels: `none` → `mild` → `moderate` → `severe`

---

#### `telemetry.py` — TelemetryCollector

Anonymous opt-in telemetry system with GDPR-compliant privacy guarantees:

- ✅ Query text is **never** sent — only routing outcomes
- ✅ User IDs are **never** sent
- ✅ Opt-in only (`ADAPTROUTER_TELEMETRY=true`)
- ✅ Confidence values bucketed to 0.1 ranges for privacy
- ✅ Timestamps rounded to hour level
- ✅ Users can clear their telemetry queue at any time

---

#### `config.py` — Configuration

All tunable parameters in one place:

| Parameter | Default | Description |
|---|---|---|
| `RETRAIN_THRESHOLD` | `20` | Minimum feedback examples before retraining |
| `MIN_RETRAIN_INTERVAL_HOURS` | `1.0` | Minimum hours between retraining events |
| `DRIFT_THRESHOLD` | `0.15` | L2 distance threshold for drift alerts |
| `ACCURACY_TOLERANCE` | `0.02` | Max accuracy drop allowed during retraining |
| `SHAP_DIRECTION_THRESHOLD` | `0.05` | SHAP score threshold for word direction labels |
| `LONG_QUERY_WORD_LIMIT` | `150` | Queries above this word count → auto-complex |
| `CALIBRATION_MIN_BUCKET_SIZE` | `3` | Minimum samples per calibration bucket |
| `REPHRASING_SIMILARITY_THRESHOLD` | `0.82` | Cosine similarity threshold for rephrasing detection |
| `REPHRASING_TIME_WINDOW_SECONDS` | `60` | Time window for rephrasing detection |
| `FAST_MODEL` | `llama-3.1-8b-instant` | Groq model ID for simple queries |
| `SMART_MODEL` | `llama-3.3-70b-versatile` | Groq model ID for complex queries |

---

## 🖥️ Dashboard

The Streamlit dashboard (`app.py`) provides a full interactive interface:

### Features

- **Query input** with model routing and answer display
- **Routing metadata pills** — label, confidence, P(simple), P(complex), latency, tokens
- **SHAP explanations** — expandable panel showing why each model was chosen, with word-level complexity chips
- **Adaptive answers** — if you rate an answer as unhelpful and describe the issue, re-asking the same question generates an improved response using your feedback note
- **Feedback system** — 👍/👎 buttons, 1–5 star rating, free-text feedback notes
- **Learning progress** — sidebar showing feedback counts, retrain countdown, classifier accuracy, feedback health
- **Conversation history** — last 8 queries with full metadata, feedback chains for iterative improvement

### Adaptive Answer Flow

1. Ask a question → get answer
2. Click 👎 → describe what was wrong
3. Ask the **same question** again
4. AdaptRouter detects the similarity (Jaccard ≥ 55% word overlap)
5. Builds an improved prompt incorporating your feedback note
6. Returns a significantly better answer marked with ♻️ **Adapted**

---

## 🧪 Testing

The project includes comprehensive unit tests for all modules:

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_core.py -v
pytest tests/test_explainer.py -v
pytest tests/test_calibration.py -v
pytest tests/test_feedback_store.py -v
pytest tests/test_feedback_collector.py -v
pytest tests/test_implicit_feedback.py -v
pytest tests/test_retrainer.py -v
pytest tests/test_telemetry.py -v
```

### Test Coverage

| Module | Test File | Coverage |
|---|---|---|
| `core.py` | `test_core.py` | AdaptRouter init, classify, route, feedback, stats |
| `explainer.py` | `test_explainer.py` | SHAP explanations, word scores, batch explain |
| `calibration.py` | `test_calibration.py` | ECE computation, calibration curves |
| `feedback_store.py` | `test_feedback_store.py` | SQLite CRUD, stats, retrain logging |
| `feedback_collector.py` | `test_feedback_collector.py` | Explicit/implicit recording, retrain triggers |
| `implicit_feedback.py` | `test_implicit_feedback.py` | Rephrasing, escalation, acceptance detection |
| `retrainer.py` | `test_retrainer.py` | Warm-start training, validation, acceptance/rejection |
| `telemetry.py` | `test_telemetry.py` | Event recording, flushing, privacy guarantees |

---

## 📊 Benchmarks

### Running Benchmarks

```bash
# Full comparison: static vs adaptive router
python benchmark/final_comparison.py

# Quick comparison with 20 queries
python benchmark/compare_routers.py

# Simulate 3 rounds of feedback-driven improvement
python benchmark/simulate_feedback.py

# SHAP explainer demo
python benchmark/day7_explainer_demo.py

# Calibration analysis demo
python benchmark/day8_calibration_demo.py

# Performance profiling (requires Ollama running locally)
python benchmark/profiler.py
```

### Benchmark Results

Results from a 30-query benchmark (queries not seen during training):

| Metric | llm-router (static) | **AdaptRouter** (after retrain) |
|---|---|---|
| Routing accuracy | Baseline | +improvement after 20 signals |
| Queries to fast model | Varies | Optimized |
| Latency saving vs always-smart | ~22% | ~22%+ |
| Cost saving vs always-smart | ~91% | ~91%+ |
| Calibration error (ECE) | Measured | Improved with Platt scaling |
| Drift detection | ❌ | ✅ Active |
| Domain adaptation | ❌ | ✅ Automatic |
| Self-retraining | ❌ | ✅ After 20 signals |
| Explainability (SHAP) | ❌ | ✅ Word-level |

### Domain Adaptation Example

```bash
# Run the medical domain adaptation demo
python -m adaptrouter.examples.medical_domain
```

Demonstrates how the router adapts to medical queries that were **not** in the original CS/ML training data — purely through user feedback.

---

## 🐳 Docker

```bash
# Build the Docker image
docker build -t adaptrouter .

# Run the container
docker run -p 7860:7860 --env-file .env adaptrouter
```

The container runs the Streamlit dashboard on port `7860`.

---

## 📁 Project Structure

```
adaptrouter/
├── adaptrouter/                   # Core Python package
│   ├── __init__.py                # Package init — exports AdaptRouter
│   ├── core.py                    # Main AdaptRouter class
│   ├── config.py                  # All configuration parameters
│   ├── feedback_store.py          # SQLite persistence layer
│   ├── feedback_collector.py      # Feedback orchestration (Facade pattern)
│   ├── implicit_feedback.py       # Behavioural signal detection
│   ├── retrainer.py               # Warm-start retraining + Platt calibration
│   ├── explainer.py               # SHAP-based routing explanations
│   ├── calibration.py             # Confidence calibration analysis (ECE)
│   ├── drift_detector.py          # Distribution drift detection
│   ├── telemetry.py               # Anonymous opt-in telemetry
│   └── examples/
│       └── medical_domain.py      # Domain adaptation example
├── dashboard/
│   └── app.py                     # Streamlit dashboard (alternate entry)
├── benchmark/
│   ├── compare_routers.py         # Static vs adaptive comparison
│   ├── final_comparison.py        # Paper-ready 30-query benchmark
│   ├── simulate_feedback.py       # 3-round feedback simulation
│   ├── profiler.py                # Inference latency profiler
│   ├── day7_explainer_demo.py     # SHAP explainer demo
│   └── day8_calibration_demo.py   # Calibration analysis demo
├── tests/                         # Unit tests for all modules
│   ├── test_core.py
│   ├── test_explainer.py
│   ├── test_calibration.py
│   ├── test_feedback_store.py
│   ├── test_feedback_collector.py
│   ├── test_implicit_feedback.py
│   ├── test_retrainer.py
│   └── test_telemetry.py
├── data/
│   └── adaptrouter.db             # SQLite database (auto-created)
├── app.py                         # Main Streamlit dashboard entry point
├── pyproject.toml                 # Package metadata and dependencies
├── requirements.txt               # Pinned dependencies
├── Dockerfile                     # Docker container configuration
├── runtime.txt                    # Python version for deployment
├── .env                           # Environment variables (not in git)
└── .gitignore
```

---

## ⚙️ Optional Dependencies

Install only what you need:

```bash
# Explainability (SHAP word-level explanations)
pip install -e ".[explainability]"

# Dashboard (Streamlit + Plotly)
pip install -e ".[dashboard]"

# Development (pytest, build tools)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

---

## 🛣️ Roadmap

- [ ] Multi-provider support (OpenAI, Anthropic, Ollama)
- [ ] A/B testing framework for routing strategies
- [ ] REST API endpoint for microservice deployment
- [ ] Embedding cache for repeated queries
- [ ] Active learning: prioritise uncertain queries for feedback
- [ ] Per-user routing profiles

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👤 Author

**Shreya** — [GitHub](https://github.com/shreyagobi)

---

<div align="center">

*AdaptRouter v0.1.0 — Self-improving LLM routing middleware*

</div>
