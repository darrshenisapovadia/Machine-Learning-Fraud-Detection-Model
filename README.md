
# Three-Model Deep Learning Fraud Detection System

## Overview
This repository contains a production-ready fraud detection system built using multiple deep learning models and exposed through a FastAPI-based REST API.
The system detects fraudulent financial transactions in real time by combining supervised classification and unsupervised anomaly detection techniques.

The project is designed to be modular, testable, and deployment-ready, making it suitable for real-world applications, learning, and portfolio use.

Video Demo: https://drive.google.com/file/d/18t6wdQ1ZDQvwjHDykfMJAb7XA9NbcMye/view?usp=sharing

---

## System Architecture
The system uses three independent deep learning models and an ensemble strategy:

1. MLP (Multi-Layer Perceptron)
   - Supervised neural network
   - Outputs fraud probability between 0 and 1

2. Autoencoder (AE)
   - Unsupervised anomaly detector
   - Uses reconstruction error for fraud detection

3. Variational Autoencoder (VAE)
   - Probabilistic anomaly detection model
   - Detects rare and statistically abnormal transactions

### Ensemble Logic
- Predictions from all three models are combined
- Majority voting is used (2 out of 3)
- Improves reliability and reduces false positives

---

## Key Features
- Real-time fraud prediction using FastAPI
- Individual endpoints for each model
- Ensemble fraud prediction endpoint
- Consistent preprocessing across training and inference
- Threshold-based anomaly detection
- Interactive HTML dashboard
- Comprehensive unit and integration tests

---

## Project Structure

├── app.py                     # FastAPI application
├── models/                    # Model definitions
│   ├── mlp.py
│   ├── autoencoder.py
│   ├── vae.py
│   └── __init__.py
├── saved_models/              # Trained models and artifacts
│   ├── mlp_model_tuned.pth
│   ├── autoencoder_model_tuned.pth
│   ├── vae_model_tuned.pth
│   ├── scaler.pkl
│   └── thresholds.json
├── notebooks/                 # Model training notebooks
├── tests/                     # Unit and integration tests
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

---

## Input Data Format
The API expects transaction data with 30 features:

- Time
- V1 to V28 (PCA-transformed features)
- Amount

All inputs must be numeric and follow the same format used during training.

---

## API Endpoints

### Health and Metadata
- GET /
- GET /health
- GET /models

### Prediction Endpoints
- POST /predict/mlp
- POST /predict/autoencoder
- POST /predict/vae
- POST /predict/ensemble

### Dashboard
- GET /dashboard

---

## Example Request

{
  "Time": 1000,
  "V1": 0.1,
  "V2": -0.2,
  "V3": 0.05,
  "V4": 0.01,
  "V5": -0.03,
  "V6": 0.2,
  "V7": 0,
  "V8": 0,
  "V9": 0,
  "V10": 0,
  "V11": 0,
  "V12": 0,
  "V13": 0,
  "V14": 0,
  "V15": 0,
  "V16": 0,
  "V17": 0,
  "V18": 0,
  "V19": 0,
  "V20": 0,
  "V21": 0,
  "V22": 0,
  "V23": 0,
  "V24": 0,
  "V25": 0,
  "V26": 0,
  "V27": 0,
  "V28": 0,
  "Amount": 250
}

---

## Example Response (Ensemble)

{
  "model": "Ensemble",
  "ensemble_is_fraud": false,
  "mlp_probability": 0.47,
  "mlp_is_fraud": false,
  "ae_anomaly_score": 0.41,
  "ae_is_fraud": false,
  "vae_anomaly_score": 0.45,
  "vae_is_fraud": false
}

---

## How to Run the Project

1. Clone Repository
git clone https://github.com/darrshenisapovadia/Machine-Learning-Fraud-Detection-Model.git
cd Machine-Learning-Fraud-Detection-Model

2. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Start FastAPI Server
uvicorn app:app --reload

5. Access Application
Swagger UI: http://127.0.0.1:8000/docs
Dashboard: http://127.0.0.1:8000/dashboard

---

## Testing
The tests folder is independent of the FastAPI runtime.

It is used for:
- Model architecture validation
- Model weight compatibility checks
- API endpoint testing
- End-to-end pipeline validation
- Performance and latency testing

Run tests using:
pytest tests/

---

## Use Cases
- Credit card fraud detection
- Financial risk monitoring
- Machine learning system design
- AI/ML portfolio projects
- Research and experimentation

---

## Author
Darrsheni Sapovadia
