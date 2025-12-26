# check_model_architectures.py
"""
Utility script to check saved model architectures and compare them 
against the current model definitions.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.vae import VAE

print(" Checking saved model architectures...\n")

def compare_state_dicts(saved_state, current_state, model_name):
    """Compare saved state_dict with current model's state_dict"""
    print(f"\n=== {model_name} ===")
    
    saved_keys = set(saved_state.keys())
    current_keys = set(current_state.keys())
    
    missing_keys = current_keys - saved_keys
    unexpected_keys = saved_keys - current_keys
    
    if missing_keys:
        print(f" Missing keys in saved state ({len(missing_keys)}): {list(missing_keys)}")
    if unexpected_keys:
        print(f" Unexpected keys in saved state ({len(unexpected_keys)}): {list(unexpected_keys)}")
    
    # Compare parameter shapes
    for key in saved_keys & current_keys:
        saved_shape = tuple(saved_state[key].shape)
        current_shape = tuple(current_state[key].shape)
        if saved_shape != current_shape:
            print(f" Shape mismatch at '{key}': saved {saved_shape}, expected {current_shape}")
        else:
            print(f" {key} shape matches: {saved_shape}")

# Check MLP
try:
    mlp_data = torch.load('saved_models/mlp_model.pth', map_location='cpu')
    mlp_model = MLP(input_dim=30)
    if isinstance(mlp_data, dict):
        print("MLP: State dict loaded.")
        compare_state_dicts(mlp_data, mlp_model.state_dict(), "MLP")
    else:
        print("MLP: Complete model object saved (not just state_dict).")
        print(mlp_data)
except Exception as e:
    print(f"Error loading MLP: {e}")

# Check Autoencoder
try:
    ae_data = torch.load('saved_models/autoencoder_model.pth', map_location='cpu')
    ae_model = Autoencoder(input_dim=30)
    if isinstance(ae_data, dict):
        print("\nAutoencoder: State dict loaded.")
        compare_state_dicts(ae_data, ae_model.state_dict(), "Autoencoder")
    else:
        print("\nAutoencoder: Complete model object saved (not just state_dict).")
        print(ae_data)
except Exception as e:
    print(f"Error loading Autoencoder: {e}")

# Check VAE
try:
    vae_data = torch.load('saved_models/vae_model.pth', map_location='cpu')
    vae_model = VAE(input_dim=30)
    if isinstance(vae_data, dict):
        print("\nVAE: State dict loaded.")
        compare_state_dicts(vae_data, vae_model.state_dict(), "VAE")
    else:
        print("\nVAE: Complete model object saved (not just state_dict).")
        print(vae_data)
except Exception as e:
    print(f"Error loading VAE: {e}")
print("\n Model architecture check complete.")