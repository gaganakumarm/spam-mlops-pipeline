# 🔬 Advanced System Design & MLOps Analysis

## 📈 Performance Analysis
*Measured on a standard 4-core CPU environment.*
- **Heuristic Latency:** ~2-5ms.
- **ML Inference Latency:** ~12-18ms.
- **System Throughput:** Optimized for high-concurrency via FastAPI's asynchronous event loop.

## 🔄 MLOps Lifecycle & Governance
We utilize **MLflow** not just as a logger, but as a **Model Registry**:
1. **Experimentation:** Tracking F1-scores to prevent model degradation.
2. **Promotion:** Only models tagged `Production` are pulled by the API, decoupling data science experiments from production stability.



## 🚦 Automated Quality Gates (CI/CD)
The pipeline is designed to include:
- **Accuracy Thresholds:** Automated checks to block models with < 90% accuracy.
- **Environment Parity:** Using Docker to ensure the "Training Environment" matches the "Inference Environment" exactly.

## 🔮 Future Roadmap & Scalability
- **Monitoring:** Integrating Prometheus to track "Spam vs Ham" ratios in real-time.
- **Drift Detection:** Automated alerts when the incoming text distribution shifts significantly from the training set.
- **Distributed Serving:** Horizontal scaling of the FastAPI container behind an Nginx load balancer.