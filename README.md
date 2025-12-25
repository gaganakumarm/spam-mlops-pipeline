# Hybrid Spam Detection Pipeline

## Overview

A hybrid spam classification service that combines rule-based heuristics with a machine-learning model to deliver low-latency, reliable inference.
The system is production-structured, versioned with MLflow, and containerized for reproducible deployment.

---

## Problem

Pure ML-based spam filters:

* Add unnecessary latency for obvious cases
* Can fail silently on rule-obvious inputs
* Are harder to reason about in production

This system addresses those issues by introducing a **deterministic heuristic layer** ahead of ML inference.

---

## Architecture

```
Client
  ↓
FastAPI Inference Service
  ↓
Heuristic Filter ──► Spam (early exit)
  ↓
TF-IDF Vectorizer
  ↓
Logistic Regression Model
  ↓
Prediction
```

**Key properties**

* Early-exit path for obvious spam
* ML inference only when required
* Clear separation between inference, model lifecycle, and deployment

---

## Model & Experiments

Experiments are tracked using MLflow to ensure reproducibility and controlled promotion.

* **Model:** Logistic Regression
* **Features:** TF-IDF (word n-grams 1–2)
* **Best Configuration:** C = 1.0
* **Accuracy:** 98.2%
* **Deployment Policy:** Only models promoted to `Production` are loaded by the API

> Training and inference are fully decoupled via the MLflow Model Registry.

---

## Performance Characteristics

Measured on a standard local CPU environment.

* **Heuristic path latency:** ~2–5 ms
* **ML inference latency:** ~12–18 ms
* **Concurrency:** FastAPI async request handling

These are **expected runtime characteristics**, not formal load-test benchmarks.

---

## Tech Stack

* **Language:** Python 3.9+
* **ML:** Scikit-learn
* **API:** FastAPI
* **MLOps:** MLflow
* **Infra:** Docker, Docker Compose

---

## Running the Service

```bash
docker-compose up
```

API documentation:

```
http://localhost:8000/docs
```

---

## Design Rationale: Hybrid Inference

Not all inputs require statistical modeling.

**Advantages**

* Reduced average latency
* Deterministic behavior for known patterns
* Lower compute cost under load

**Limitations**

* Heuristic rules are static
* Offset by ML handling ambiguous cases

---

## Future Work

* Model drift detection on incoming text distribution
* Prometheus-based metrics for inference monitoring
* Horizontal scaling behind a reverse proxy

---

## Documentation

* System design and MLOps lifecycle:
  [`docs/SYSTEM_DESIGN.md`](./docs/SYSTEM_DESIGN.md)

