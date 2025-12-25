# Hybrid Spam Detection Pipeline (MLOps)

A production-grade text classification service utilizing a **Hybrid Heuristic-ML architecture**. This system optimizes for the "Three Pillars of MLOps": **Latency, Governance, and Reproducibility.**

## One-Line Value Proposition
A "Fail-Fast" inference engine that catches 80% of spam via heuristics in <2ms, delegating nuanced cases to a versioned Scikit-Learn model managed via MLflow.

## System Architecture
The system is designed as a modular microservice, decoupling the inference logic from the model lifecycle.



- **FastAPI Layer:** Asynchronous request handling for high concurrency.
- **Heuristic Filter:** Deterministic regex-based early exit.
- **ML Registry:** Hot-swappable model artifacts via MLflow.

## Performance & Results
| Metric | Value | Context |
| :--- | :--- | :--- |
| **P99 Latency (Heuristic)** | **~2ms** | String matching / Regex |
| **P99 Latency (ML Path)** | **~18ms** | TF-IDF + LogReg |
| **Model Accuracy** | **98.2%** | Validation on balanced dataset |
| **Governance** | `Production` | Tagged in MLflow Registry |

## Quick Start
1. Ensure Docker Desktop is running.
2. Run `./run_demo.bat` (Windows) or `docker-compose up`.
3. Access the interactive API docs at `http://localhost:8000/docs`.

---
**Engineering Deep Dive:** See [**docs/SYSTEM_DESIGN.md**](./docs/SYSTEM_DESIGN.md) for architectural trade-offs, MLOps governance, and scalability roadmaps.
