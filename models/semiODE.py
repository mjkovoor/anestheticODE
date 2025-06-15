import torch
import torch.nn as nn

class SemiMechanisticODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_c = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.k_decay = nn.Parameter(torch.tensor(0.5))  # learnable

    def forward(self, t, x):
        R = x[:, 0:1]
        C = x[:, 1:2]
        dR_dt = self.f_c(C) - self.k_decay * R
        return dR_dt