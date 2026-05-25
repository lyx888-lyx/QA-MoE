import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_labels)
        )

    def forward(self, x):
        return self.fc(x)  # [B, num_labels]
