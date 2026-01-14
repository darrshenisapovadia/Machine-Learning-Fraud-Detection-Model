# test_mlp_architecture.py
import torch
import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP

print("✓ Successfully imported MLP model")

# Initialize model
model = MLP(input_dim=30)
print("\nModel architecture:\n", model)

# Load saved weights
saved_path = "saved_models/mlp_model.pth"
if not os.path.exists(saved_path):
    raise FileNotFoundError(f"Could not find {saved_path}")

saved_state = torch.load(saved_path, map_location="cpu")
model.load_state_dict(saved_state)
print("\n✓ Successfully loaded weights into model")

# ---- Training mode test ----
model.train()
train_input = torch.randn(4, 30)  # batch size > 1 required for BatchNorm
with torch.no_grad():
    output_train = model(train_input)
print(f"✓ Training mode forward pass works! Output shape: {output_train.shape}")

# ---- Evaluation mode test ----
model.eval()
test_input = torch.randn(1, 30)  # single sample
with torch.no_grad():
    output_eval = model(test_input)
print(f"✓ Eval mode forward pass works! Output: {output_eval.item():.6f}")
