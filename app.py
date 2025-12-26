#!/usr/bin/env python3
import os
import json
import joblib
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch

# Import models (your newly created files)
from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.vae import VAE

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection ML Model",
    version="1.0.0",
    description="""
An end-to-end, production-ready fraud detection platform built using multiple deep learning architectures.
This API evaluates financial transactions in real time and determines fraud risk using both supervised
and unsupervised learning techniques.

### Machine Learning Engines
The system exposes predictions from three independently trained neural networks:

- **Multi-Layer Perceptron (MLP)**  
  A supervised neural network trained on labelled transaction data to directly estimate fraud probability.

- **Autoencoder (AE)**  
  An unsupervised reconstruction-based model that identifies abnormal transaction patterns by measuring
  reconstruction error.

- **Variational Autoencoder (VAE)**  
  A probabilistic anomaly detection model that captures latent transaction distributions and flags
  statistically rare behaviour.

### Core Capabilities
- Dedicated REST endpoints for each model as well as a majority-voting ensemble
- Automatic feature ordering, scaling, and preprocessing identical to the training pipeline
- Configurable threshold-based anomaly scoring for unsupervised models
- Lightweight HTML dashboard for quick manual testing without external tools
- Designed for easy integration with banking systems, dashboards, and monitoring pipelines

This API is suitable for research, prototyping, and real-time fraud detection use cases.
"""
,
    contact={
        "name": "Darrsheni Sapovadia",
        "url": "https://darrshenisapovadia.github.com"
    }
)

# -------------------------------------------------------------------
# Paths (based on YOUR folder structure screenshot)
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Your screenshot shows scaler.pkl + thresholds.json inside saved_models
ARTIFACTS_DIR = os.path.join(BASE_DIR, "saved_models")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
THRESHOLDS_JSON = os.path.join(ARTIFACTS_DIR, "thresholds.json")

# Prefer tuned weights if present
MLP_TUNED_PATH = os.path.join(MODELS_DIR, "mlp_model_tuned.pth")
AE_TUNED_PATH  = os.path.join(MODELS_DIR, "autoencoder_model_tuned.pth")
VAE_TUNED_PATH = os.path.join(MODELS_DIR, "vae_model_tuned.pth")

MLP_PATH = os.path.join(MODELS_DIR, "mlp_model.pth")
AE_PATH  = os.path.join(MODELS_DIR, "autoencoder_model.pth")
VAE_PATH = os.path.join(MODELS_DIR, "vae_model.pth")


def pick_path(tuned_path: str, normal_path: str) -> str:
    """Choose tuned model if available, else normal."""
    return tuned_path if os.path.exists(tuned_path) else normal_path


# -------------------------------------------------------------------
# Helpers (robust loading)
# -------------------------------------------------------------------
def load_state_dict_any(path: str) -> dict:
    """
    Loads a PyTorch .pth that could be:
    - raw state_dict
    - checkpoint dict with keys like 'state_dict' / 'model_state_dict'
    """
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        # if it already looks like state_dict
        if any(isinstance(k, str) and (k.endswith(".weight") or k.endswith(".bias")) for k in obj.keys()):
            return obj

    return obj


# -------------------------------------------------------------------
# Load artifacts + models at startup
# -------------------------------------------------------------------
logger.info("Loading scaler + thresholds + models...")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"scaler.pkl not found at: {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)
logger.info("✓ Scaler loaded")

# thresholds.json (your screenshot shows keys: ae_threshold, vae_threshold, mlp_threshold)
thresholds = {"ae_threshold": None, "vae_threshold": None, "mlp_threshold": 0.5}
if os.path.exists(THRESHOLDS_JSON):
    with open(THRESHOLDS_JSON, "r") as f:
        thresholds.update(json.load(f))
    logger.info("✓ Thresholds loaded from thresholds.json")
else:
    logger.warning(f"thresholds.json not found at: {THRESHOLDS_JSON} (will use defaults where possible)")

# Validate threshold keys
if thresholds.get("ae_threshold") is None or thresholds.get("vae_threshold") is None:
    logger.warning("ae_threshold / vae_threshold missing in thresholds.json. Autoencoder/VAE anomaly decisions may be incorrect.")

# Pick model files
mlp_file = pick_path(MLP_TUNED_PATH, MLP_PATH)
ae_file  = pick_path(AE_TUNED_PATH, AE_PATH)
vae_file = pick_path(VAE_TUNED_PATH, VAE_PATH)

logger.info(f"Using MLP weights: {os.path.basename(mlp_file)}")
logger.info(f"Using AE  weights: {os.path.basename(ae_file)}")
logger.info(f"Using VAE weights: {os.path.basename(vae_file)}")

# Instantiate
mlp_model = MLP(input_dim=30)
ae_model = Autoencoder(input_dim=30, latent_dim=4)
vae_model = VAE(input_dim=30, latent_dim=8)

# Load weights (strict=False so minor layer name differences won't crash)
mlp_state = load_state_dict_any(mlp_file)
missing, unexpected = mlp_model.load_state_dict(mlp_state, strict=False)
mlp_model.eval()
logger.info("✓ MLP loaded")
if missing:
    logger.warning(f"MLP missing keys (ignored): {missing}")
if unexpected:
    logger.warning(f"MLP unexpected keys (ignored): {unexpected}")

ae_state = load_state_dict_any(ae_file)
ae_model.load_state_dict(ae_state, strict=False)
ae_model.eval()
logger.info("✓ Autoencoder loaded")

vae_state = load_state_dict_any(vae_file)
vae_model.load_state_dict(vae_state, strict=False)
vae_model.eval()
logger.info("✓ VAE loaded")


# -------------------------------------------------------------------
# Request Schema
# -------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5",
    "V6", "V7", "V8", "V9", "V10", "V11",
    "V12", "V13", "V14", "V15", "V16", "V17",
    "V18", "V19", "V20", "V21", "V22", "V23",
    "V24", "V25", "V26", "V27", "V28", "Amount"
]


def preprocess(transaction: Transaction) -> np.ndarray:
    x = np.array([[getattr(transaction, f) for f in FEATURE_ORDER]], dtype=np.float32)
    return scaler.transform(x)


def predict_mlp(x_scaled: np.ndarray):
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        out = mlp_model(x_tensor)

        # out could be shape [1,1] sigmoid OR raw logits
        val = out.view(-1).item()

        # If it's already between 0..1, treat as probability
        if 0.0 <= val <= 1.0:
            prob = float(val)
        else:
            prob = float(torch.sigmoid(torch.tensor(val)).item())

    threshold = float(thresholds.get("mlp_threshold", 0.5))
    return prob, prob > threshold


def predict_autoencoder(x_scaled: np.ndarray):
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        recon = ae_model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1).item()

    thr = thresholds.get("ae_threshold")
    is_fraud = bool(error > float(thr)) if thr is not None else False
    return float(error), is_fraud


def predict_vae(x_scaled: np.ndarray):
    with torch.no_grad():
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        recon, _, _ = vae_model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1).item()

    thr = thresholds.get("vae_threshold")
    is_fraud = bool(error > float(thr)) if thr is not None else False
    return float(error), is_fraud


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {"message": "Fraud Detection API is running! Visit /docs or /dashboard."}


@app.get("/health", tags=["General"])
def health_check():
    return {"status": "healthy"}


@app.get("/models", tags=["Models"])
def list_models():
    return {"models": ["mlp", "autoencoder", "vae", "ensemble"]}


@app.post("/predict/mlp", tags=["Predictions"])
def predict_with_mlp(transaction: Transaction):
    try:
        x_scaled = preprocess(transaction)
        prob, is_fraud = predict_mlp(x_scaled)
        return {"model": "MLP", "probability": prob, "threshold": thresholds.get("mlp_threshold", 0.5), "is_fraud": bool(is_fraud)}
    except Exception as e:
        logger.exception("MLP prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/autoencoder", tags=["Predictions"])
def predict_with_autoencoder(transaction: Transaction):
    try:
        x_scaled = preprocess(transaction)
        error, is_fraud = predict_autoencoder(x_scaled)
        return {"model": "Autoencoder", "anomaly_score": error, "threshold": thresholds.get("ae_threshold"), "is_fraud": bool(is_fraud)}
    except Exception as e:
        logger.exception("Autoencoder prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/vae", tags=["Predictions"])
def predict_with_vae(transaction: Transaction):
    try:
        x_scaled = preprocess(transaction)
        error, is_fraud = predict_vae(x_scaled)
        return {"model": "VAE", "anomaly_score": error, "threshold": thresholds.get("vae_threshold"), "is_fraud": bool(is_fraud)}
    except Exception as e:
        logger.exception("VAE prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ensemble", tags=["Predictions"])
def predict_ensemble(transaction: Transaction):
    try:
        x_scaled = preprocess(transaction)
        mlp_prob, mlp_fraud = predict_mlp(x_scaled)
        ae_error, ae_fraud = predict_autoencoder(x_scaled)
        vae_error, vae_fraud = predict_vae(x_scaled)

        votes = [mlp_fraud, ae_fraud, vae_fraud]
        ensemble_fraud = sum(votes) >= 2

        return {
            "model": "Ensemble",
            "ensemble_is_fraud": bool(ensemble_fraud),
            "mlp_probability": mlp_prob,
            "mlp_is_fraud": bool(mlp_fraud),
            "ae_anomaly_score": ae_error,
            "ae_is_fraud": bool(ae_fraud),
            "vae_anomaly_score": vae_error,
            "vae_is_fraud": bool(vae_fraud),
        }
    except Exception as e:
        logger.exception("Ensemble prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Minimal HTML Dashboard
# -------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            input { width: 80px; margin: 2px; }
            button { margin-top: 10px; padding: 8px 12px; }
            #result { margin-top: 20px; font-weight: bold; white-space: pre; }
        </style>
    </head>
    <body>
        <h1>Fraud Detection Dashboard</h1>
        <h5> By: Darrsheni Sapovadia </h5>
        <p>Quick test: enter Time, V1..V6, Amount. Remaining features will be set to 0.</p>

        <form id="fraudForm">
            <div>Time: <input name="Time" value="0"></div>
            <div>
                V1: <input name="V1" value="0">
                V2: <input name="V2" value="0">
                V3: <input name="V3" value="0">
            </div>
            <div>
                V4: <input name="V4" value="0">
                V5: <input name="V5" value="0">
                V6: <input name="V6" value="0">
            </div>
            <div>Amount: <input name="Amount" value="0"></div>

            <button type="button" onclick="predict()">Predict (Ensemble)</button>
        </form>

        <div id="result"></div>

        <script>
            async function predict() {
                const form = document.getElementById('fraudForm');
                const data = {};
                for (let el of form.elements) {
                    if (el.name) data[el.name] = parseFloat(el.value);
                }
                const rest = ["V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"];
                rest.forEach(f => data[f] = 0);

                const res = await fetch('/predict/ensemble', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });

                const json = await res.json();
                document.getElementById('result').innerText = JSON.stringify(json, null, 2);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# -------------------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Fraud Detection API...")
    logger.info("Run: python -m uvicorn app:app --reload")
    logger.info("Docs: http://127.0.0.1:8000/docs")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
