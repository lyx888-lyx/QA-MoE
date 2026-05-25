import torch
import torch.nn as nn

from model.encoders.MultiModalEncoder import TextEncoder, AudioEncoder, VisionEncoder
from model.moe.MoEBlock import MoEBlock
from model.heads.RegressionHead import RegressionHead
from model.heads.ClassificationHead import ClassificationHead
from model.modules.transformer import TransformerEncoder


class MoEMultimodalEmotionModel(nn.Module):
    """
    Final Missing-Modality Aware MoE Model
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.shared_dim = getattr(config, "shared_dim", 256)

        self.is_classification = config.dataset == "MIntRec"
        if self.is_classification:
            self.num_labels = config.num_labels

        # ----------------------------------------------------
        # 1. Encoders (all return [B,T,H])
        # ----------------------------------------------------
        self.text_encoder = TextEncoder(finetune=config.finetune_bert)
        self.audio_encoder = AudioEncoder(config.audio_input_dim, self.hidden_dim)
        self.vision_encoder = VisionEncoder(config.vision_input_dim, self.hidden_dim)

        # Temporal context
        self.text_transformer = TransformerEncoder(
            embed_dim=self.hidden_dim, num_heads=4, layers=1,
            attn_dropout=0.1, relu_dropout=0.1,
            res_dropout=0.1, embed_dropout=0.1, attn_mask=False
        )

        # Router 输入 = pooled concatenation
        router_input_dim = self.hidden_dim * 3

        # ----------------------------------------------------
        # 2. MoE Block
        # ----------------------------------------------------
        self.moe = MoEBlock(
            input_dim=router_input_dim,
            expert_dim=self.shared_dim,
            num_experts=config.num_experts,
            router_hidden_dim=128,
            top_k=config.top_k,
            use_quality=config.use_quality,
            text_dim=self.hidden_dim,
            audio_dim=self.hidden_dim,
            vision_dim=self.hidden_dim
        )

        # ----------------------------------------------------
        # 3. Regression Head
        # ----------------------------------------------------
        if self.is_classification:
            self.head = ClassificationHead(self.shared_dim, self.num_labels)
        else:
            self.head = RegressionHead(self.shared_dim)

    def forward(self, text_input, audio_input, vision_input,
                missing_text, missing_audio, missing_vision):

        # --------------------------------------------------------
        # 1. Encode with missing mask
        # --------------------------------------------------------
        # print("text_input.shape:", text_input.shape) # torch.Size([16, 3, 50])
        # print("missing_text.shape:", missing_text.shape) # torch.Size([16])
        text_feat = self.text_encoder(text_input, missing_text)      # [16,3,50]
        audio_feat = self.audio_encoder(audio_input, missing_audio)  # [B,50,128]
        vision_feat = self.vision_encoder(vision_input, missing_vision)  # [B,50,128]

        # --------------------------------------------------------
        # 2. Transformer per modality
        # --------------------------------------------------------
        text_feat = self.text_transformer(text_feat)
        # audio_feat = self.audio_transformer(audio_feat)
        # vision_feat = self.vision_transformer(vision_feat)
        
        def apply_clean_mask(feat, mask):
            if mask is None: return feat
            # mask: [B] -> [B, 1, 1] 用于广播乘法
            m = (~mask).float().unsqueeze(1).unsqueeze(2) 
            return feat * m  # 强制将缺失模态的 Transformer Bias 归零

        text_feat   = apply_clean_mask(text_feat, missing_text)
        audio_feat  = apply_clean_mask(audio_feat, missing_audio)
        vision_feat = apply_clean_mask(vision_feat, missing_vision)

        # --------------------------------------------------------
        # 3. Pooled feature for router
        # --------------------------------------------------------
        pooled_feature = torch.cat([
            text_feat.mean(dim=1),
            audio_feat.mean(dim=1),
            vision_feat.mean(dim=1)
        ], dim=-1)

        pooled_feature = torch.nan_to_num(pooled_feature, nan=0.0, posinf=1e4, neginf=-1e4)

        # --------------------------------------------------------
        # 4. MoE Block WITH modality_mask
        # --------------------------------------------------------
        fused_z, gating_weights, quality_signal, entropy_loss, modality_mask = self.moe(
            pooled_feature,
            {
                'text': text_feat,
                'audio': audio_feat,
                'vision': vision_feat
            },
            modality_mask={
                'text': missing_text,
                'audio': missing_audio,
                'vision': missing_vision
            }
        )

        # --------------------------------------------------------
        # 5. Regression head
        # --------------------------------------------------------
        output = self.head(fused_z)

        return {
            'output': output.squeeze(-1),
            'fused_z': fused_z,
            'gating_weights': gating_weights,
            'quality_signal': quality_signal,
            'modality_mask': modality_mask,
            'entropy_loss': entropy_loss
        }
