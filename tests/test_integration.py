#!/usr/bin/env python3


import unittest
import numpy as np
import torch
import joblib
import pickle
import os
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.vae import VAE


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete fraud detection system"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests"""
        print("\nSetting up integration test environment...")

        # Load test data
        cls.X_test = np.load("saved_artifacts/X_test.npy")
        cls.y_test = np.load("saved_artifacts/y_test.npy")

        # Load scaler
        cls.scaler = joblib.load("saved_artifacts/scaler.pkl")

        # Load thresholds
        try:
            with open("saved_artifacts/threshold_ae.pkl", "rb") as f:
                cls.ae_threshold = pickle.load(f)
            with open("saved_artifacts/threshold_vae.pkl", "rb") as f:
                cls.vae_threshold = pickle.load(f)
        except:
            cls.ae_threshold = 1.0
            cls.vae_threshold = 1.0

        # Load models
        input_dim = cls.X_test.shape[1]

        cls.mlp_model = MLP(input_dim=input_dim)
        cls.mlp_model.load_state_dict(torch.load("saved_models/mlp_model.pth", map_location="cpu"))
        cls.mlp_model.eval()

        cls.ae_model = Autoencoder(input_dim=input_dim)
        cls.ae_model.load_state_dict(torch.load("saved_models/autoencoder_model.pth", map_location="cpu"))
        cls.ae_model.eval()

        cls.vae_model = VAE(input_dim=input_dim)
        cls.vae_model.load_state_dict(torch.load("saved_models/vae_model.pth", map_location="cpu"))
        cls.vae_model.eval()

        print("Integration test environment setup complete!\n")

    def test_data_loading(self):
        """Test that test data is loaded correctly"""
        self.assertIsNotNone(self.X_test, "X_test should not be None")
        self.assertIsNotNone(self.y_test, "y_test should not be None")
        self.assertEqual(self.X_test.shape[0], self.y_test.shape[0], "X_test and y_test should have same number of samples")
        self.assertEqual(self.X_test.shape[1], 30, "X_test should have 30 features")

    def test_scaler_loading(self):
        """Test that scaler is loaded and functional"""
        self.assertIsNotNone(self.scaler, "Scaler should not be None")

        sample_data = self.X_test[:5]
        scaled_data = self.scaler.transform(sample_data)

        self.assertEqual(scaled_data.shape, sample_data.shape, "Scaled data should have same shape as input")
        self.assertFalse(np.any(np.isnan(scaled_data)), "Scaled data should not contain NaN values")

    def test_model_loading(self):
        """Test that all models are loaded correctly"""
        models = [self.mlp_model, self.ae_model, self.vae_model]
        model_names = ["MLP", "Autoencoder", "VAE"]

        for model, name in zip(models, model_names):
            self.assertIsNotNone(model, f"{name} model should not be None")
            params = list(model.parameters())
            self.assertGreater(len(params), 0, f"{name} should have parameters")

    def test_mlp_prediction(self):
        """Test MLP model prediction pipeline"""
        sample_data = self.X_test[:3]
        scaled_data = self.scaler.transform(sample_data)
        tensor_data = torch.FloatTensor(scaled_data)

        with torch.no_grad():
            predictions = self.mlp_model(tensor_data)
            pred_probs = predictions.numpy().flatten()
            pred_labels = (pred_probs > 0.5).astype(int)

        self.assertEqual(len(pred_probs), len(sample_data))
        self.assertTrue(np.all((pred_probs >= 0) & (pred_probs <= 1)))

    def test_autoencoder_anomaly_detection(self):
        """Test Autoencoder anomaly detection pipeline"""
        sample_data = self.X_test[:3]
        scaled_data = self.scaler.transform(sample_data)
        tensor_data = torch.FloatTensor(scaled_data)

        with torch.no_grad():
            _, errors = self.ae_model.reconstruct(tensor_data)
            errors = errors.numpy()
            anomalies = (errors > self.ae_threshold).astype(int)

        self.assertEqual(len(errors), len(sample_data))
        self.assertTrue(np.all(errors >= 0))

    def test_vae_anomaly_detection(self):
        """Test VAE anomaly detection pipeline"""
        sample_data = self.X_test[:3]
        scaled_data = self.scaler.transform(sample_data)
        tensor_data = torch.FloatTensor(scaled_data)

        with torch.no_grad():
            _, errors, _, _ = self.vae_model.reconstruct(tensor_data)
            errors = errors.numpy()
            anomalies = (errors > self.vae_threshold).astype(int)

        self.assertEqual(len(errors), len(sample_data))
        self.assertTrue(np.all(errors >= 0))

    def test_threshold_application(self):
        """Test that thresholds are applied correctly"""
        test_errors = np.array([0.5, 1.0, 1.5, 2.0])
        ae_predictions = (test_errors > self.ae_threshold).astype(int)
        vae_predictions = (test_errors > self.vae_threshold).astype(int)

        self.assertEqual(len(ae_predictions), len(test_errors))
        self.assertEqual(len(vae_predictions), len(test_errors))

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from raw data to prediction"""
        sample_indices = np.random.choice(len(self.X_test), 5, replace=False)
        sample_data = self.X_test[sample_indices]
        sample_labels = self.y_test[sample_indices]

        scaled_data = self.scaler.transform(sample_data)
        tensor_data = torch.FloatTensor(scaled_data)

        with torch.no_grad():
            mlp_probs = self.mlp_model(tensor_data).numpy().flatten()
            mlp_preds = (mlp_probs > 0.5).astype(int)

            _, ae_errors = self.ae_model.reconstruct(tensor_data)
            ae_preds = (ae_errors.numpy() > self.ae_threshold).astype(int)

            _, vae_errors, _, _ = self.vae_model.reconstruct(tensor_data)
            vae_preds = (vae_errors.numpy() > self.vae_threshold).astype(int)

        self.assertEqual(len(mlp_preds), len(sample_data))
        self.assertEqual(len(ae_preds), len(sample_data))
        self.assertEqual(len(vae_preds), len(sample_data))

        print("\nEnd-to-End Sample Results:")
        print(f"Labels: {sample_labels}")
        print(f"MLP: {mlp_preds}")
        print(f"AE: {ae_preds}")
        print(f"VAE: {vae_preds}")

    def test_ensemble_prediction(self):
        """Test ensemble prediction logic (mirrors FastAPI /predict/ensemble)"""
        sample_indices = np.random.choice(len(self.X_test), 5, replace=False)
        sample_data = self.X_test[sample_indices]

        scaled_data = self.scaler.transform(sample_data)
        tensor_data = torch.FloatTensor(scaled_data)

        with torch.no_grad():
            # MLP
            mlp_probs = self.mlp_model(tensor_data).numpy().flatten()
            mlp_preds = (mlp_probs > 0.5).astype(int)

            # AE
            _, ae_errors = self.ae_model.reconstruct(tensor_data)
            ae_errors = ae_errors.numpy()
            ae_preds = (ae_errors > self.ae_threshold).astype(int)

            # VAE
            _, vae_errors, _, _ = self.vae_model.reconstruct(tensor_data)
            vae_errors = vae_errors.numpy()
            vae_preds = (vae_errors > self.vae_threshold).astype(int)

        # Ensemble = majority voting
        ensemble_preds = (mlp_preds + ae_preds + vae_preds >= 2).astype(int)

        self.assertEqual(len(ensemble_preds), len(sample_data))
        self.assertTrue(np.all((ensemble_preds == 0) | (ensemble_preds == 1)))

        print("\nEnsemble Prediction Results:")
        print(f"MLP: {mlp_preds}")
        print(f"AE: {ae_preds}")
        print(f"VAE: {vae_preds}")
        print(f"Ensemble: {ensemble_preds}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
