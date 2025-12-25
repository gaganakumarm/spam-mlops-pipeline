# System Design & MLOps Analysis

##  1. Performance Characteristics

*Estimates derived from local containerized environments using FastAPI's asynchronous profiling.*

| Execution Path | Latency (P99) | Mechanism |
| --- | --- | --- |
| **Heuristic Path** | **~2–5 ms** | Deterministic Regex / String Matching |
| **ML Inference Path** | **~12–18 ms** | TF-IDF + Logistic Regression |
| **Concurrency** | **High** | Python Asyncio Event Loop |

---

##  2. Hybrid Inference Architecture

The system employs a **Fail-Fast Layered Defense** to optimize for both speed and compute cost.

### Inference Workflow:

1. **Ingress:** RESTful request enters via FastAPI.
2. **Layer 1 (Heuristic):** A deterministic filter scans for high-confidence spam patterns. If a match is found, the system exits early with a classification, bypassing the ML stack entirely.
3. **Layer 2 (ML Inference):** If Layer 1 is inconclusive, the payload is vectorized via TF-IDF and processed by a Logistic Regression classifier.
4. **Resolution:** The system fetches the active model artifact aliased as `Production` from the MLflow Registry, ensuring the API is decoupled from the training environment.

---

##  3. MLOps Governance & Lifecycle

We utilize **MLflow as a Control Plane** to manage the model lifecycle from experimentation to retirement.

* **Experimentation:** Hyperparameter tuning results (C-values, n-grams) are logged to ensure 100% reproducibility.
* **Model Registry:** Provides a centralized source of truth. Models are versioned and require an explicit manual or automated "Promotion" to the `Production` stage.
* **Artifact Isolation:** Inference code never points to a local `.pkl` file; it queries the registry, allowing for hot-swapping models without restarting the service.

---

##  4. Quality Gates & Environment Parity

To maintain high reliability, the system is designed with "Guardrails":

* **Environment Pinning:** Docker ensures that the `scikit-learn` and `python` binaries used during training are bit-for-bit identical to those in production.
* **Validation Gating:** The architecture supports CI/CD hooks (e.g., GitHub Actions) that prevent the promotion of any model that fails to meet a minimum **F1-Score threshold (e.g., >0.95)**.

---

##  5. Architectural Trade-offs

| Decision | Pros | Cons |
| --- | --- | --- |
| **Logistic Regression** | Low latency, highly interpretable, low memory footprint. | Limited ability to capture complex semantic context compared to Transformers. |
| **Hybrid Logic** | Massive CPU savings on obvious spam; deterministic control. | Increased code complexity; requires maintaining both rules and models. |
| **SQLite/Local MLflow** | Zero-config, easy to demo and test locally. | Not suitable for multi-node distributed training (would require Postgres/S3). |

---

##  6. Scalability Roadmap

* **Observability:** Integration of **Prometheus** and **Grafana** to monitor the "Heuristic vs. ML" hit ratio.
* **Drift Detection:** Implementing Kolmogorov-Smirnov tests to detect when the statistical distribution of incoming text shifts away from training data.
* **Horizontal Scaling:** Orchestrating FastAPI replicas via **Kubernetes (K8s)** with an Nginx Ingress Controller.

