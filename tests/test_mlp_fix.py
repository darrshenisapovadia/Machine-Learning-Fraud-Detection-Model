# test_mlp_fix.py
import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now try to import
try:
    from models.mlp import MLP
    print("✓ Successfully imported MLP model")
    
    # Create model with new architecture
    model = MLP(input_dim=30)
    print("New model architecture:")
    print(model)
    
    # Try loading the saved weights
    try:
        saved_state = torch.load('saved_models/mlp_model.pth')
        model.load_state_dict(saved_state)
        print("✓ Successfully loaded weights!")
        
        # Test forward pass
        test_input = torch.randn(1, 30)
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass works! Output: {output.item()}")
        
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        
        # Check what keys are missing/unexpected
        current_state = model.state_dict()
        
        print("\nMissing keys in current model:")
        for key in saved_state.keys():
            if key not in current_state:
                print(f"  - {key}")
        
        print("\nUnexpected keys in saved state:")
        for key in current_state.keys():
            if key not in saved_state:
                print(f"  - {key}")
                
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    if os.path.exists('models'):
        print("Files in models directory:", os.listdir('models'))
    else:
        print("Models directory does not exist!")