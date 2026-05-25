import torch
import torch.nn as nn
import torch.nn.functional as F

from model.moe.ExpertBank import ExpertBank
from model.moe.GatingRouter import GatingRouter
from model.moe.MoEAggregator import MoEAggregator
from model.moe.QualityPredictor import QualityPredictor


class MoEInputProjector(nn.Module):
    """
    Project [B, T, D] → [B, shared_dim]
    Performs temporal average pooling + linear projection.
    """
    def __init__(self, input_dim, shared_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)

    def forward(self, x):
        x = torch.nan_to_num(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        return self.norm(torch.nan_to_num(x))


class MoEBlock(nn.Module):
    """
    Missing-Modality Aware MoE
    """
    def __init__(
        self,
        input_dim,
        expert_dim=256,
        num_experts=8,
        router_hidden_dim=128,
        top_k=2,
        use_quality=True,
        text_dim=128,
        audio_dim=128,
        vision_dim=128
    ):
        super().__init__()

        self.use_quality = use_quality
        self.num_experts = num_experts
        self.shared_dim = expert_dim
        self.top_k = top_k

        # 1) Quality Predictors
        if use_quality:
            self.q_text = QualityPredictor(text_dim)
            self.q_audio = QualityPredictor(audio_dim)
            self.q_vision = QualityPredictor(vision_dim)

        # 2) Projectors
        self.text_proj   = MoEInputProjector(text_dim,   shared_dim=expert_dim)
        self.audio_proj  = MoEInputProjector(audio_dim,  shared_dim=expert_dim)
        self.vision_proj = MoEInputProjector(vision_dim, shared_dim=expert_dim)

        # 3) Expert Bank
        self.expert_bank = ExpertBank(num_experts, expert_dim)

        # 4) Router
        self.router = GatingRouter(
            input_dim=input_dim,
            num_experts=num_experts,
            num_modalities=3
        )

        # 5) Aggregator
        self.aggregator = MoEAggregator(expert_dim, top_k, fallback_idx=num_experts - 1)

        # 6) 分别处理质量阈值与 missing mask
        self.quality_threshold = 0.3

    def forward(self, pooled_feature, modality_inputs, modality_mask):
        """
        modality_inputs = {"text":..., "audio":..., "vision":...}
        modality_mask   = {"text": [B], "audio": [B], "vision": [B]}
        """
        pooled_feature = torch.nan_to_num(pooled_feature)

        text_seq   = torch.nan_to_num(modality_inputs["text"])
        audio_seq  = torch.nan_to_num(modality_inputs["audio"])
        vision_seq = torch.nan_to_num(modality_inputs["vision"])

        mt = modality_mask["text"]   # [B]
        ma = modality_mask["audio"]
        mv = modality_mask["vision"]

        # ==============================================
        # Step 1. Quality prediction (only for PRESENT modalities)
        # ==============================================
        if self.use_quality:
            q_t = self.q_text(text_seq)
            q_a = self.q_audio(audio_seq)
            q_v = self.q_vision(vision_seq)
            
            # [B, 3] 范围 [0, 1]
            raw_quality = torch.cat([q_t, q_a, q_v], dim=1)
            quality_signal = torch.sigmoid(raw_quality) # 确保严格在0-1之间且平滑

            # --- 关键修改：处理物理缺失 ---
            # 如果物理上就缺失 (modality_mask=1)，则强制 Quality=0
            # 使用乘法而不是 if-else，保持梯度流
            # mask: 1=missing, 0=present
            physical_presence = 1.0 - torch.stack([mt, ma, mv], dim=1).float()
            quality_signal = quality_signal * physical_presence
            
            # --- 关键修改：生成 Soft Missing Mask ---
            # 不要用 < 0.3 的硬截断，而是用 (1 - q) 作为缺失程度
            # q越小，soft_mask越大，Fallback bias 越强
            soft_modality_mask = 1.0 - quality_signal 

        else:
            quality_signal = torch.ones(text_seq.size(0), 3).to(text_seq.device)
            soft_modality_mask = torch.zeros_like(quality_signal)


        # ==============================================
        # Step 2. Router with missing mask
        # ==============================================
        gating_weights, entropy_loss = self.router(
            pooled_feature=pooled_feature,
            quality_signal=quality_signal
        )

        # ==============================================
        # Step 3. Project to shared expert space

        text_feat   = self.text_proj(text_seq)
        audio_feat  = self.audio_proj(audio_seq)
        vision_feat = self.vision_proj(vision_seq)

        # --- 关键修改：基于 Quality 的加权输入 ---
        # 如果 Vision 质量很差 (q_v=0.1), 它对 expert_input 的贡献应该很小
        # 这样专家看到的输入更“纯净”
        
        q_t_broad = quality_signal[:, 0:1]
        q_a_broad = quality_signal[:, 1:2]
        q_v_broad = quality_signal[:, 2:3]
        
        # 分母加 epsilon 防止除零
        weighted_sum = (text_feat * q_t_broad + audio_feat * q_a_broad + vision_feat * q_v_broad)
        sum_q = (q_t_broad + q_a_broad + q_v_broad).clamp(min=1e-6)
        
        expert_input = weighted_sum / sum_q
        expert_input = torch.nan_to_num(expert_input)
        # ==============================================
        # Step 4. Expert outputs
        # ==============================================
        expert_outputs = []
        for i in range(self.num_experts):
            z = self.expert_bank(i, expert_input)
            expert_outputs.append(torch.nan_to_num(z))

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, N, D]

        # ==============================================
        # Step 5. Aggregator
        # ==============================================
        fused = self.aggregator(expert_outputs, gating_weights, modality_mask=None) 

        return fused, gating_weights, quality_signal, entropy_loss, soft_modality_mask

