# 🛡️ Hybrid Spam Detection Pipeline (MLOps)

## 🚀 One-Line Value Proposition
A production-ready spam classification service using a hybrid heuristic-ML approach, managed with MLflow and containerized with Docker.

## ⚠️ The Problem
Standard ML models can be "black boxes" that fail on obvious cases or require high compute. This project provides a **reliable** alternative that combines human-defined rules with statistical learning.

## 🏗️ Modular Architecture
- **Inference Service:** FastAPI wrapper providing REST endpoints.
- **Hybrid Logic:** Heuristic keyword filtering followed by a TF-IDF Logistic Regression model.
- **Model Management:** MLflow for tracking experiments and versioning models.
- **Orchestration:** Docker Compose for seamless, one-click deployment.



## 🛠️ Tech Stack
- **Languages:** Python 3.9+
- **Frameworks:** Scikit-Learn, FastAPI
- **MLOps:** MLflow
- **Infrastructure:** Docker, Docker Compose

## 🚀 Quick Start
1. Ensure Docker is running.
2. Run `./run_demo.bat` (Windows) or `docker-compose up`.
3. Open `http://localhost:8000/docs` to test the API.

## 🧠 Key Design Decision: Why Hybrid?
Instead of sending every request to the ML model, we use a **Heuristic Layer** first. 
- **Pros:** Sub-millisecond response for obvious spam; deterministic behavior.
- **Cons:** Rules are static. (Solved by the ML layer handling the nuances).

---
📖 **For a deep dive into benchmarks, MLOps lifecycle, and future scalability, see [SYSTEM_DESIGN.md](./docs/SYSTEM_DESIGN.md).**
