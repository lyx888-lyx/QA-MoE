import torch
import torch.nn as nn

class RegressionMoELoss(nn.Module):
    """
    MOSI/MOSEI Loss: Regression + Contrastive + MoE Reg + 🔥Quality Supervision
    """
    def __init__(self, reg_weight=1.0, quality_weight=0.5, entropy_weight=0.01):
        super().__init__()
        self.reg_weight = reg_weight
        self.quality_weight = quality_weight # 新增：质量损失的权重
        self.entropy_weight = entropy_weight # 建议：调小一点，0.1可能太大
        
        self.l1loss = nn.L1Loss()
        # 用于监督 quality_signal (输出是0-1之间的标量)
        self.quality_loss_fn = nn.MSELoss() 

    def forward(self, outputs, labels):
        """
        outputs: 包含了 model 输出的所有内容
        labels: 
            {
                "reg": [B], Ground Truth
                "missing_mask": { "text": [B], "audio": [B], "vision": [B] } 
                (注意：你需要确保 dataloader 传进来这个 mask)
            }
        """

        reg_pred = outputs["output"].squeeze()
        reg_true = labels["reg"]

        # ==========================================
        # 1) Main Loss: Regression
        # ==========================================
        loss_reg = self.l1loss(reg_pred, reg_true)
        total_loss = self.reg_weight * loss_reg

        # ==========================================
        # 2) 🔥 Critical Addition: Quality Loss
        # ==========================================
        # 我们必须教 QualityPredictor：缺失的时候输出0，存在的时候输出1
        if "quality_signal" in outputs:
            # pred_q: [B, 3] (Text, Audio, Vision)
            pred_q = outputs["quality_signal"]
            
            # 构建 target_q (Ground Truth for Quality)
            # labels["missing_mask"] 是 1=Missing, 0=Present
            # 所以 Quality Target 应该是 1 - mask
            # 确保 mask 在同一个 device 上
            
            m_text = labels["missing_mask"]["text"]
            m_audio = labels["missing_mask"]["audio"]
            m_vision = labels["missing_mask"]["vision"]
            
            # 拼接成 [B, 3]
            missing_stack = torch.stack([m_text, m_audio, m_vision], dim=1).float()
            
            # Target: 存在=1.0, 缺失=0.0
            target_q = 1.0 - missing_stack 
            
            # MSE Loss: 让预测的 q 逼近真实的物理存在情况
            loss_q = self.quality_loss_fn(pred_q, target_q)
            
            total_loss += self.quality_weight * loss_q
        else:
            loss_q = torch.tensor(0.0)

        # ==========================================
        # 3) MoE Entropy Regularization
        # ==========================================
        # 建议把权重从 0.1 降到 0.01 或 0.02
        # 因为 Reg Loss 通常在 0.7 左右，如果 Entropy 也是 0.X，会干扰主任务
        if "entropy_loss" in outputs and outputs["entropy_loss"] is not None:
            total_loss += self.entropy_weight * outputs["entropy_loss"]

        # ==========================================
        # 4) Contrastive (Optional)
        # ==========================================
        if "contrastive_loss" in outputs and outputs["contrastive_loss"] is not None:
             total_loss += 0.05 * outputs["contrastive_loss"]

        return total_loss, {
            "loss_reg": loss_reg.item(),
            "loss_quality": loss_q.item(),
            "total_loss": total_loss.item()
        }