"""
Unit tests for the deep learning models in the fraud detection system.
Tests MLP, Autoencoder, and VAE models individually and for consistency.
"""

import unittest
import torch
import numpy as np
import os
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.vae import VAE


class TestMLP(unittest.TestCase):
    """Unit tests for the MLP model"""

    def setUp(self):
        torch.manual_seed(42)
        self.input_dim = 30
        self.batch_size = 32
        self.mlp_model = MLP(input_dim=self.input_dim)
        self.dummy_input = torch.randn(self.batch_size, self.input_dim)

    def test_mlp_initialization(self):
        self.assertIsNotNone(self.mlp_model)
        parameters = list(self.mlp_model.parameters())
        self.assertGreater(len(parameters), 0)
        self.assertEqual(self.mlp_model.model[-1].__class__.__name__, 'Sigmoid')

    def test_mlp_forward_pass(self):
        with torch.no_grad():
            output = self.mlp_model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")

    def test_mlp_training_mode(self):
        self.assertTrue(self.mlp_model.training)
        self.mlp_model.eval()
        self.assertFalse(self.mlp_model.training)
        self.mlp_model.train()
        self.assertTrue(self.mlp_model.training)

    def test_mlp_dropout(self):
        self.mlp_model.train()
        out_train = self.mlp_model(self.dummy_input)
        self.mlp_model.eval()
        out_eval = self.mlp_model(self.dummy_input)
        self.assertFalse(torch.allclose(out_train, out_eval))


class TestAutoencoder(unittest.TestCase):
    """Unit tests for the Autoencoder model"""

    def setUp(self):
        torch.manual_seed(42)
        self.input_dim = 30
        self.batch_size = 32
        self.ae_model = Autoencoder(input_dim=self.input_dim)
        self.dummy_input = torch.randn(self.batch_size, self.input_dim)

    def test_autoencoder_initialization(self):
        self.assertIsNotNone(self.ae_model.encoder)
        self.assertIsNotNone(self.ae_model.decoder)
        self.assertGreater(len(list(self.ae_model.parameters())), 0)

    def test_autoencoder_forward_pass(self):
        with torch.no_grad():
            reconstructed = self.ae_model(self.dummy_input)
        self.assertEqual(reconstructed.shape, self.dummy_input.shape)

    def test_autoencoder_reconstruct_method(self):
        with torch.no_grad():
            reconstructed, error = self.ae_model.reconstruct(self.dummy_input)
        self.assertEqual(reconstructed.shape, self.dummy_input.shape)
        self.assertEqual(error.shape, (self.batch_size,))
        self.assertTrue(torch.all(error >= 0))

    def test_autoencoder_encoding(self):
        with torch.no_grad():
            encoded = self.ae_model.encoder(self.dummy_input)
        self.assertEqual(encoded.shape, (self.batch_size, 8))


class TestVAE(unittest.TestCase):
    """Unit tests for the VAE model"""

    def setUp(self):
        torch.manual_seed(42)
        self.input_dim = 30
        self.batch_size = 32
        self.vae_model = VAE(input_dim=self.input_dim)
        self.dummy_input = torch.randn(self.batch_size, self.input_dim)

    def test_vae_initialization(self):
        self.assertIsNotNone(self.vae_model.encoder)
        self.assertIsNotNone(self.vae_model.decoder)
        self.assertIsNotNone(self.vae_model.fc_mu)
        self.assertIsNotNone(self.vae_model.fc_var)

    def test_vae_forward_pass(self):
        with torch.no_grad():
            reconstructed, mu, log_var = self.vae_model(self.dummy_input)
        self.assertEqual(reconstructed.shape, self.dummy_input.shape)
        self.assertEqual(mu.shape, (self.batch_size, 8))
        self.assertEqual(log_var.shape, (self.batch_size, 8))

    def test_vae_encode_method(self):
        with torch.no_grad():
            mu, log_var = self.vae_model.encode(self.dummy_input)
        self.assertEqual(mu.shape, (self.batch_size, 8))
        self.assertEqual(log_var.shape, (self.batch_size, 8))

    def test_vae_reparameterize(self):
        mu = torch.randn(10, 8)
        log_var = torch.randn(10, 8)
        z = self.vae_model.reparameterize(mu, log_var)
        self.assertEqual(z.shape, mu.shape)

    def test_vae_loss_function(self):
        rec = torch.randn(10, self.input_dim)
        orig = torch.randn(10, self.input_dim)
        mu = torch.randn(10, 8)
        log_var = torch.randn(10, 8)
        loss = self.vae_model.loss_function(rec, orig, mu, log_var)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)

    def test_vae_reconstruct_method(self):
        with torch.no_grad():
            reconstructed, error, mu, log_var = self.vae_model.reconstruct(self.dummy_input)
        self.assertEqual(reconstructed.shape, self.dummy_input.shape)
        self.assertEqual(error.shape, (self.batch_size,))
        self.assertTrue(torch.all(error >= 0))


class TestModelConsistency(unittest.TestCase):
    """Consistency checks across all models"""

    def test_input_dimension_consistency(self):
        input_dim = 30
        mlp = MLP(input_dim)
        ae = Autoencoder(input_dim)
        vae = VAE(input_dim)

        dummy_input = torch.randn(5, input_dim)
        with torch.no_grad():
            mlp_output = mlp(dummy_input)
            ae_output = ae(dummy_input)
            vae_output, _, _ = vae(dummy_input)

        self.assertEqual(mlp_output.shape[0], 5)
        self.assertEqual(ae_output.shape[0], 5)
        self.assertEqual(vae_output.shape[0], 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
