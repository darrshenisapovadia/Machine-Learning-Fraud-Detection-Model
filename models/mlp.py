import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=30, hidden=(64, 32, 16), dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.BatchNorm1d(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden[0], hidden[1]),
            nn.BatchNorm1d(hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden[1], hidden[2]),
            nn.BatchNorm1d(hidden[2]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
