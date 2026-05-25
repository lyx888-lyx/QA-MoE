import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super(RegressionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1)
        )


    def forward(self, x):
        return self.fc(x).squeeze(-1) # [B, 1] → [B]