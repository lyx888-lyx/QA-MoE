import torch
import torch.nn as nn


class MoEAggregator(nn.Module):
    """
    Robust Top-k Aggregator for Missing-Modality MoE
    """

    def __init__(self, expert_dim, top_k=2, fallback_idx=None):
        super().__init__()
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.fallback_idx = fallback_idx

    def forward(self, expert_outputs, gating_weights, modality_mask=None):
        """
        expert_outputs: list of [B, D] tensors or single tensor [B, N, D]
        gating_weights: [B, N]
        modality_mask:  [B, 3] (optional)
        return: fused [B, D]
        """
        # --- 自动识别输入类型 ---
        if isinstance(expert_outputs, (list, tuple)):
            stacked = torch.stack(expert_outputs, dim=1)  # [B, N, D]
        else:
            stacked = expert_outputs  # 已经是 tensor [B, N, D]

        stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
        gating_weights = torch.nan_to_num(gating_weights, nan=0.0)

        # --- top-k mask ---
        if self.top_k is not None and self.top_k < stacked.size(1):
            topk_vals, topk_idx = torch.topk(gating_weights, self.top_k, dim=-1)
            mask = torch.zeros_like(gating_weights).scatter_(1, topk_idx, 1.0)
            gating_weights = gating_weights * mask
            gating_weights = gating_weights / gating_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # --- 加权聚合 ---
        fused = torch.bmm(gating_weights.unsqueeze(1), stacked).squeeze(1)  # [B, D]
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- optional mask penalty (for debugging stability) ---
        if modality_mask is not None:
            modality_mask = torch.nan_to_num(modality_mask, nan=0.0)
        
        return fused

