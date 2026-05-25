import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.contrastive_loss import EmotionContrastiveLoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)          # higher pt = easier sample
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction=='mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        cls_weight=0.2,
        reg_weight=0.6,
        contrastive_weight=0.2,
        use_focal=False,
        gamma=2.0,
    ):
        super().__init__()

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.contrastive_weight = contrastive_weight

        self.use_focal = use_focal
        self.gamma = gamma

        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.reg = nn.MSELoss()

    def focal_loss(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

    def forward(
        self,
        model_outputs,
        labels,
        epoch=None,
        contrastive_loss_fn=None
    ):
        logits = model_outputs['class_logits']
        reg_pred = model_outputs['regression'].view(-1)  # safe reshape

        cls_target = labels['cls'].long()
        reg_target = labels['reg'].view(-1)

        # ---- classification ----
        if self.use_focal:
            loss_cls = self.focal_loss(logits, cls_target)
        else:
            loss_cls = self.ce(logits, cls_target)

        # ---- regression ----
        loss_reg = self.reg(reg_pred, reg_target)

        # ---- base total ----
        total = (
            self.cls_weight * loss_cls +
            self.reg_weight * loss_reg
        )

        # ---- contrastive loss (由 trainer 注入 fn) ----
        if (
            contrastive_loss_fn is not None and
            self.contrastive_weight > 0
        ):
            z = model_outputs['fused_z']

            # compute contrastive loss
            contrastive_loss = contrastive_loss_fn(z, reg_target)

            # warm-up
            if epoch is not None and epoch < 3:
                w = 0.03 * self.contrastive_weight
            else:
                w = self.contrastive_weight

            total += w * contrastive_loss
        else:
            contrastive_loss = torch.tensor(0.0)

        return total, {
            "loss_cls": loss_cls.item(),
            "loss_reg": loss_reg.item(),
            "loss_contrast": contrastive_loss.item(),
            "total_loss": total.item()
        }
