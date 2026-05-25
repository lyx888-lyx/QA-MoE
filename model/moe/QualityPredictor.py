import torch
import torch.nn as nn

class QualityPredictor(nn.Module):
    """
    Predicts modality quality score q ∈ [0,1]
    Input:  [B, T, D]
    Output: [B, 1]
    """
    def __init__(self, input_dim):
        super().__init__()
        
        # We input 2 * D because we concatenate Mean + Std
        self.fc1 = nn.Linear(input_dim * 2, input_dim) 
        self.fc2 = nn.Linear(input_dim, 1)

        # Initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        # 1. Statistical Pooling
        # Mean captures the general signal content
        mu = x.mean(dim=1) # [B, D]
        # Std captures the noise/instability (Crucial for detecting blur/noise)
        sigma = x.std(dim=1) # [B, D]
        
        # Concatenate: [B, 2D]
        stats = torch.cat([mu, sigma], dim=1)
        # ablation study: only use mean
        # fake_std = torch.zeros_like(sigma)
        # stats = torch.cat([mu, fake_std], dim=1)

        # 2. Estimate Quality
        h = torch.relu(self.fc1(stats))
        q = torch.sigmoid(self.fc2(h)) # [B, 1]

        # 3. Optional: Hard-mask for completely missing input (Safety Lock)
        # Using a slightly larger epsilon for stability
        is_present = (x.abs().sum(dim=(1, 2)) > 1e-5).float().unsqueeze(1)
        q = q * is_present

        return q