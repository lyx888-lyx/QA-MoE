import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for sentiment regression.
    Samples with similar sentiment values (labels) should get closer.
    """

    def __init__(self, temperature=0.3):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, y):
        """
        z: [B, D] fused embedding
        y: [B] sentiment regression labels (float)
        """
        B, D = z.size()

        # Normalize embeddings
        z = F.normalize(z, p=2, dim=1)

        # Pairwise cosine similarity [B,B]
        sim = torch.matmul(z, z.T) / self.temperature

        # Build supervised "positive mask" based on label closeness
        # Two samples are considered similar if |y_i - y_j| < 0.5
        y = y.view(-1, 1)                 # [B,1]
        label_diff = torch.abs(y - y.T)   # [B,B]

        pos_mask = (label_diff < 0.5).float()
        neg_mask = 1 - pos_mask

        # Remove self-contrast (i == j)
        diag = torch.eye(B, device=z.device)
        pos_mask = pos_mask * (1 - diag)

        # Positive pairs: logsumexp over negative similarities
        exp_sim = torch.exp(sim)

        # denominator = sum over all except self
        denom = exp_sim * (1 - diag)
        denom = denom.sum(dim=1, keepdim=True) + 1e-9

        # numerator = sum over positives
        pos_exp = exp_sim * pos_mask
        numerator = pos_exp.sum(dim=1, keepdim=True) + 1e-9

        # contrastive loss
        loss = -torch.log(numerator / denom)
        return loss.mean()
