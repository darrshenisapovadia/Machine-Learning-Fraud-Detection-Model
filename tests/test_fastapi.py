#!/usr/bin/env python3
"""
Test script for the FastAPI Fraud Detection Application
Author: Nafisa Lawal Idris
Description: Tests all API endpoints with sample transactions
"""

import requests
import time
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"

def wait_for_api(max_retries=10, delay=2):
    """Wait for API readiness"""
    print("Waiting for API to start...")
    for i in range(max_retries):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print("✓ API is ready!")
                return True
        except requests.exceptions.RequestException:
            print(f"  Attempt {i+1}/{max_retries}: not ready yet...")
            time.sleep(delay)
    print("✗ API failed to start in time")
    return False


def test_health_endpoints():
    """Check root, health, and models endpoints"""
    print("Testing health endpoints...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"✓ Root endpoint: {r.status_code}")

        r = requests.get(f"{BASE_URL}/health", timeout=10)
        data = r.json()
        print(f"✓ Health endpoint: {r.status_code} - Status: {data.get('status','unknown')}")

        r = requests.get(f"{BASE_URL}/models", timeout=10)
        print(f"✓ Models endpoint: {r.status_code}")
        return True
    except Exception as e:
        print(f"✗ Health endpoints failed: {e}")
        return False


def create_test_transactions():
    """Two example transactions"""
    normal_tx = {
        "Time": 406.0,
        "V1": -2.312226542, "V2": 1.951992011, "V3": -1.609850732,
        "V4": 3.997905588, "V5": -0.522187865, "V6": -1.426545319,
        "V7": -2.537387306, "V8": 1.391657248, "V9": -2.770089277,
        "V10": -2.772272145, "V11": 3.202033207, "V12": -2.899907389,
        "V13": -0.595221881, "V14": -4.289253782, "V15": 0.38972412,
        "V16": -1.14074718, "V17": -2.830055674, "V18": -0.016822468,
        "V19": 0.416955705, "V20": 0.126910559, "V21": 0.517232371,
        "V22": -0.035049369, "V23": -0.465211076, "V24": 0.320198199,
        "V25": 0.044519167, "V26": 0.177839798, "V27": 0.261145003,
        "V28": -0.143275875, "Amount": 0.0
    }
    fraud_tx = {
        "Time": 406.0,
        "V1": 1.191857111, "V2": 0.266150712, "V3": 0.166480113,
        "V4": 0.448154078, "V5": 0.060017649, "V6": -0.082360809,
        "V7": -0.078802983, "V8": 0.085101655, "V9": -0.255425128,
        "V10": -0.166974414, "V11": 1.612726661, "V12": 1.065235311,
        "V13": 0.489095016, "V14": -0.143772296, "V15": -0.635558093,
        "V16": -0.465916771, "V17": -0.166394595, "V18": -0.254515931,
        "V19": -0.272853354, "V20": -0.044078714, "V21": -0.027373976,
        "V22": -0.28692121, "V23": -0.002988816, "V24": 0.074033124,
        "V25": -0.011193966, "V26": -0.019718834, "V27": 0.002997689,
        "V28": -0.002458271, "Amount": 0.0
    }
    return normal_tx, fraud_tx


def test_individual_models(normal_tx, fraud_tx):
    """POST each tx to /predict/{model}"""
    print("\nTesting individual model endpoints...")
    models = ["mlp", "autoencoder", "vae"]
    cases = [("Normal", normal_tx), ("Fraudulent", fraud_tx)]
    all_passed = True

    for label, tx in cases:
        print(f"\n{label} Transaction:")
        for model in models:
            try:
                r = requests.post(f"{BASE_URL}/predict/{model}", json=tx, timeout=10)
                if r.status_code == 200:
                    res = r.json()
                    # Pick right score key
                    score = res.get("confidence") or res.get("anomaly_score") or res.get("score")
                    print(f"  ✓ {model.upper()}: {res['is_fraud']} (score: {score:.3f})")
                else:
                    print(f"  ✗ {model.upper()}: HTTP {r.status_code}")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ {model.upper()}: Error {e}")
                all_passed = False
    return all_passed


def test_ensemble_endpoint(normal_tx, fraud_tx):
    """Check /predict/ensemble"""
    print("\nTesting ensemble endpoint...")
    cases = [("Normal", normal_tx), ("Fraudulent", fraud_tx)]
    all_passed = True

    for label, tx in cases:
        print(f"\n{label} Transaction:")
        try:
            r = requests.post(f"{BASE_URL}/predict/ensemble", json=tx, timeout=10)
            if r.status_code == 200:
                res = r.json()
                print(f"  ✓ Ensemble: {res['ensemble_is_fraud']}")
                print(f"     MLP: {res['mlp_is_fraud']} ({res['mlp_prediction']:.3f})")
                print(f"     AE: {res['ae_is_fraud']} ({res['ae_anomaly_score']:.3f})")
                print(f"     VAE: {res['vae_is_fraud']} ({res['vae_anomaly_score']:.3f})")
            else:
                print(f"  ✗ Ensemble: HTTP {r.status_code}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Ensemble: Error {e}")
            all_passed = False
    return all_passed


def test_performance():
    """Latency check"""
    print("\nTesting performance...")
    normal_tx, _ = create_test_transactions()
    times = []
    for i in range(5):
        try:
            start = time.time()
            r = requests.post(f"{BASE_URL}/predict/mlp", json=normal_tx, timeout=5)
            end = time.time()
            if r.status_code == 200:
                ms = (end - start) * 1000
                times.append(ms)
                print(f"  Request {i+1}: {ms:.1f} ms")
        except Exception as e:
            print(f"  Request {i+1}: Error {e}")
    if times:
        avg = sum(times) / len(times)
        print(f"  Average response time: {avg:.1f} ms")
        # Allow < 2500ms for now
        return avg < 2500
    return False


def main():
    print("="*60)
    print("FASTAPI FRAUD DETECTION API TEST")
    print("="*60)

    if not wait_for_api():
        return False

    results = {
        "health_endpoints": test_health_endpoints(),
        "individual_models": test_individual_models(*create_test_transactions()),
        "ensemble_endpoint": test_ensemble_endpoint(*create_test_transactions()),
        "performance": test_performance()
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    all_ok = True
    for name, ok in results.items():
        print(f"{name:20} {'✓ PASS' if ok else '✗ FAIL'}")
        if not ok:
            all_ok = False

    print("\n" + "="*60)
    print("ALL TESTS PASSED!" if all_ok else "SOME TESTS FAILED! Please check API.")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
