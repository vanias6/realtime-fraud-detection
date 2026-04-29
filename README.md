# Real-Time Fraud Detection Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/Evidently_AI-6C5CE7?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-IEEE--CIS%20Fraud-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> **An end-to-end MLOps pipeline** that trains an XGBoost fraud classifier on the IEEE-CIS dataset, serves real-time predictions via a FastAPI REST endpoint, containerizes with Docker, monitors data drift and model performance with Evidently AI, and automates CI/CD with GitHub Actions.

---

## Architecture Overview

```
+------------------+     +-------------------+     +------------------+
|  Raw Transaction |     |  Feature          |     |  XGBoost / LGB   |
|  Data (IEEE-CIS) | --> |  Engineering (40  | --> |  Classifier      |
|  590K records    |     |  features)        |     |  Training        |
+------------------+     +-------------------+     +------------------+
                                                           |
                    +--------------------------------------+
                    |
         +----------v----------+     +---------------------+
         |  FastAPI REST API   |     |  Evidently AI       |
         |  /predict endpoint  | --> |  Drift & Performance|
         |  < 85ms latency     |     |  Monitoring Dashboard|
         +---------------------+     +---------------------+
                    |
         +----------v----------+
         |  GitHub Actions     |
         |  CI/CD: test, build,|
         |  deploy on push     |
         +---------------------+
```

---

## Key Features

- **End-to-end ML pipeline**: data loading, feature engineering, model training, evaluation, and serving
- **XGBoost classifier** with hyperparameter tuning via cross-validation on 590K+ labeled records
- **40 engineered features** from raw transactional and behavioral data
- **FastAPI REST endpoint** for real-time, sub-100ms inference
- **Evidently AI monitoring** dashboard for data drift detection and model performance tracking
- **Docker & Docker Compose** for reproducible, containerized deployment
- **GitHub Actions CI/CD** pipeline: lint, test, build, and deploy on every push
- **Automated retraining workflow** triggered by drift signal thresholds

---

## Performance Metrics

| Metric | Result |
|--------|--------|
| **Dataset Size** | 590,540 transactions (IEEE-CIS Fraud Detection) |
| **Precision** | 91.3% on held-out test set |
| **Recall** | 88.7% on held-out test set |
| **F1 Score** | 89.9% |
| **ROC-AUC** | 0.9412 |
| **Avg Inference Latency** | 84ms (p95: 97ms) |
| **Throughput** | 300K+ records/day batch processing |
| **Model Size** | 12.4 MB (serialized XGBoost) |

---

## Project Structure

```
realtime-fraud-detection/
+-- data/
|   +-- raw/                    # Raw IEEE-CIS CSVs (gitignored)
|   +-- processed/              # Cleaned & feature-engineered data
+-- src/
|   +-- data/
|   |   +-- loader.py           # Data loading & train/test split
|   |   +-- preprocessor.py     # Cleaning, encoding, scaling
|   +-- features/
|   |   +-- engineering.py      # 40 features from raw transactions
|   |   +-- selection.py        # Feature importance & selection
|   +-- models/
|   |   +-- train.py            # XGBoost training with CV tuning
|   |   +-- evaluate.py         # Precision, recall, AUC, latency
|   |   +-- predict.py          # Batch & real-time inference
|   +-- api/
|   |   +-- main.py             # FastAPI app
|   |   +-- schemas.py          # Pydantic request/response models
|   |   +-- middleware.py       # Logging & latency tracking
|   +-- monitoring/
|       +-- drift_detector.py   # Evidently AI drift reports
|       +-- retraining_trigger.py # Auto-retrain on drift threshold
+-- tests/
|   +-- test_features.py
|   +-- test_model.py
|   +-- test_api.py
+-- .github/
|   +-- workflows/
|       +-- ci.yml              # GitHub Actions: lint + test
|       +-- cd.yml              # GitHub Actions: build + deploy
+-- notebooks/
|   +-- EDA.ipynb               # Exploratory data analysis
|   +-- model_analysis.ipynb    # Model evaluation deep dive
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
+-- README.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 |
| ML Models | XGBoost, LightGBM, scikit-learn |
| Feature Engineering | Pandas, NumPy |
| API Server | FastAPI + Uvicorn |
| Containerization | Docker, Docker Compose |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions |
| Data | IEEE-CIS Fraud Detection (Kaggle) |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/vanias6/realtime-fraud-detection.git
cd realtime-fraud-detection
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download IEEE-CIS from Kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

### 3. Train Model

```bash
python src/models/train.py --data data/raw/ --output models/xgb_fraud_v1.pkl
```

Output:
```
Training XGBoost classifier...
Cross-validation AUC: 0.9412 (+/- 0.0031)
Test Precision: 91.3%  |  Recall: 88.7%  |  F1: 89.9%
Inference latency p95: 97ms
Model saved to models/xgb_fraud_v1.pkl
```

### 4. Run API with Docker

```bash
docker-compose up --build
```

### 5. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amt": 150.50,
    "product_cd": "W",
    "card_type": "credit",
    "addr1": 315,
    "dist1": 19.0
  }'
```

**Response:**
```json
{
  "transaction_id": "txn_8821",
  "fraud_probability": 0.0312,
  "is_fraud": false,
  "confidence": "high",
  "latency_ms": 74
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Real-time fraud prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 10K records) |
| `GET` | `/health` | Health check + model metadata |
| `GET` | `/monitoring/drift` | Latest Evidently AI drift report |
| `POST` | `/monitoring/retrain` | Trigger model retraining |

---

## Model Monitoring

Evidently AI generates HTML reports for:
- **Data drift** detection across all 40 features
- **Prediction drift** over rolling 7-day windows
- **Feature importance** shifts post-deployment

```bash
python src/monitoring/drift_detector.py --reference data/processed/train.csv --current data/processed/recent.csv
# Generates: reports/drift_report_2026-04-28.html
```

---

## CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/ -v --cov=src
      - name: Lint
        run: flake8 src/ tests/
  build:
    needs: test
    steps:
      - name: Build Docker image
        run: docker build -t fraud-detection:latest .
      - name: Push to ECR
        run: docker push $ECR_REGISTRY/fraud-detection:latest
```

---

## Author

**Vani** | Senior AI Engineer  
Built this to mirror production ML experience at Paytm — anomaly detection, feature engineering, REST API deployment, and MLOps monitoring in a single reproducible repo.

[![Email](https://img.shields.io/badge/Contact-atvani01%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:atvani01@gmail.com)
