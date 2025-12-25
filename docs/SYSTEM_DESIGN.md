# 🔬 Advanced System Design & MLOps Analysis

## 📈 Performance Considerations
*Estimated based on local testing and FastAPI’s documented performance characteristics.*

- **Heuristic Path Latency:** ~2–5ms (string matching only).
- **ML Inference Latency:** ~12–18ms for TF-IDF + Logistic Regression.
- **Concurrency Model:** FastAPI’s async event loop enables efficient handling of concurrent requests.

> These figures represent **expected performance characteristics**, not formal load-test benchmarks.

---

## 🔄 MLOps Lifecycle & Governance
MLflow is used as a **model lifecycle manager**, not just an experiment logger:

- **Experimentation:** Tracking accuracy and F1-score across training runs.
- **Model Registry:** Registering models and manually promoting a stable version to `Production`.
- **Decoupling:** Training iterations do not affect live inference unless explicitly promoted.

---

## 🚦 Quality Gates (Design-Level)
The system is **designed** to support automated quality checks, including:

- **Metric Thresholds:** Blocking promotion of models below a defined accuracy or F1-score.
- **Environment Parity:** Docker ensures consistency between training and inference environments.

> These checks are part of the **intended CI/CD design** and can be integrated using GitHub Actions.

---

## 🔮 Future Roadmap & Scalability
- **Monitoring:** Prometheus-based metrics for request volume and prediction distribution.
- **Drift Detection:** Alerts when incoming text distributions diverge from training data.
- **Horizontal Scaling:** Multiple FastAPI replicas behind an Nginx load balancer.
