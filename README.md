# Hybrid Spam Detection Pipeline (MLOps)

A production-ready spam classification service that combines **rule-based heuristics** with **Machine Learning** to deliver high-accuracy, low-latency inference.

## The Problem

Standard ML-only filters often suffer from:

* **Latency Bloat:** Processing obvious "WINNER!" spam through heavy models.
* **Lack of Control:** Difficulty in overriding model mistakes without retraining.
* **Silent Failures:** Ambiguity in "black-box" decision-making.

## Architecture & Logic Flow

This system utilizes a **Fail-Fast** design to minimize compute costs and maximize reliability.

1. **Heuristic Filter:** Immediate regex-based detection for "early-exit" (Spam).
2. **TF-IDF + Logistic Regression:** Statistical analysis for nuanced or ambiguous cases.
3. **Model Registry:** The Inference API dynamically pulls only the version tagged `Production`.

---

## Experimentation & Results

Managed via **MLflow**, I conducted a hyperparameter sweep across 13 runs to optimize for F1-Score and generalizability.

| Metric | Result |
| --- | --- |
| **Accuracy** | **98.2%** |
| **F1-Score** | **0.975** |
| **Best Params** | C=1.0, N-gram (1,2) |
| **Stage** | `Production` |

---

## Tech Stack

* **Frameworks:** Scikit-Learn, FastAPI
* **MLOps:** MLflow (Tracking & Registry)
* **Infrastructure:** Docker, Docker Compose
* **Language:** Python 3.12+

## Quick Start

1. **Ensure Docker is running.**
2. **Start the environment:**
```bash
# Windows
./run_demo.bat

# Linux/Mac
docker-compose up

```


3. **Test the API:** Navigate to `http://localhost:8000/docs` to use the interactive Swagger UI.

---

## Performance Characteristics

*Measured on a standard local CPU environment.*

* **Heuristic Path Latency:** ~2–5ms
* **ML Inference Latency:** ~12–18ms
* **Throughput:** Optimized via FastAPI’s asynchronous event loop.

---

## Deep Dive

For a full analysis of the system architecture, MLOps lifecycle, and future scaling strategy, see:
 **[SYSTEM_DESIGN.md](https://www.google.com/search?q=./docs/SYSTEM_DESIGN.md)**

