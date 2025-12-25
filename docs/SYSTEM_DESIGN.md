# System Design & MLOps Analysis

## Performance Characteristics

*Estimated from local testing and FastAPI’s documented runtime behavior.*

* **Heuristic path latency:** ~2–5 ms (regex / string matching only)
* **ML inference latency:** ~12–18 ms (TF-IDF + Logistic Regression)
* **Concurrency model:** Asynchronous request handling via FastAPI’s event loop

> These values represent **expected performance characteristics**, not formal load-test results.

---

## Hybrid Inference Flow

The system is designed around a **fail-fast inference strategy** to minimize average latency.

1. **Request Ingress**
   Text input is received through the FastAPI service.

2. **Layer 1 — Heuristic Filter (Deterministic)**

   * Regex-based keyword matching
   * Obvious spam is classified immediately
   * Typical execution: ~2 ms

3. **Layer 2 — ML Inference (Probabilistic)**

   * TF-IDF vectorization
   * Logistic Regression classification
   * Typical execution: ~15 ms

4. **Model Resolution**

   * The inference service dynamically loads the model tagged `Production` from the MLflow registry
   * Training artifacts are isolated from live traffic

---

## MLOps Lifecycle & Governance

MLflow is used as a **model lifecycle control plane**, not just for experiment tracking.

* **Experiment Tracking:** Accuracy and F1-score logged for each training run
* **Model Registry:** Explicit registration and manual promotion to `Production`
* **Isolation:** Training iterations cannot affect inference without an explicit promotion step

This ensures predictable behavior in production while enabling rapid experimentation.

---

## Quality Gates (Design-Level)

The system is architected to support automated validation during deployment:

* **Metric Thresholds:** Prevent promotion of models below defined accuracy or F1-score baselines
* **Environment Parity:** Docker enforces consistency between training and inference runtimes

> These checks are part of the **intended CI/CD design** and can be enforced via GitHub Actions.

---

## Scalability & Future Enhancements

* **Observability:** Prometheus metrics for request volume, latency, and prediction distribution
* **Drift Detection:** Alerts on distribution shifts between incoming data and training data
* **Horizontal Scaling:** Multiple FastAPI replicas behind an Nginx load balancer
