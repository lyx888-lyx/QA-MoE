import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingRouter(nn.Module):
    """
    Fixed GatingRouter: Multiplicative Gating & Stability Control
    """
    def __init__(self,
                 input_dim,
                 num_experts,
                 num_modalities=3,
                 router_hidden_dim=128,
                 temperature=1.0):
        super().__init__()

        self.num_experts = num_experts
        self.T = temperature

        # 1. Base Router (根据内容决定想选谁)
        self.base_router = nn.Sequential(
            nn.Linear(input_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.ReLU(),
            nn.Linear(router_hidden_dim, num_experts)
        )

        # 2. Quality Gate (学习 3个模态 -> N个专家的 乘法系数)
        # 输出范围限制在 [0, 1] 之间，作为门控信号
        self.quality_proj = nn.Linear(num_modalities, num_experts)
        
        # 初始化：让 quality gate 初始时接近 1 (也就是不干预)，慢慢学着去关闭专家
        nn.init.constant_(self.quality_proj.weight, 0.0)
        nn.init.constant_(self.quality_proj.bias, 2.0) # Sigmoid(2.0) ≈ 0.88

    def forward(self, pooled_feature, quality_signal, modality_mask=None):
        """
        注意：这里不再使用 modality_mask，因为 quality_signal 已经包含了缺失信息
        """
        
        # Step 1: Base Logits
        # [B, N]
        logits = self.base_router(pooled_feature)

        # Step 2: Quality Gating
        # quality_signal: [B, 3] -> [B, N]
        quality_gate = torch.sigmoid(self.quality_proj(quality_signal))
        gated_logits = logits * quality_gate

        # Albation Study
        # gated_logits = logits

        # Step 3: Stability (Z-Loss Calculation)
        router_z_loss = torch.mean(torch.logsumexp(gated_logits, dim=-1)**2) * 1e-2

        # Step 4: Softmax
        probs = F.softmax(gated_logits / self.T, dim=-1)

        # Step 5: Entropy Loss (防止专家坍塌，稍微留一点点)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean()
        entropy_loss = 0.0 # 先关掉，排除干扰

        # return total auxiliary loss
        aux_loss = router_z_loss + entropy_loss

        return probs, aux_loss