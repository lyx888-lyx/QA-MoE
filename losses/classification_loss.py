import torch
import torch.nn as nn

class ClassificationMoELoss(nn.Module):
    """
    分类任务（MIntRec）的损失：
        - 主损失：CrossEntropy
        - 可选：MoE entropy regularization（熵正则）
        - 可选：MoE load-balance loss（平衡损失）

    输出格式与 RegressionMoELoss 完全一致：
        return total_loss, {
            "loss_cls": ...,
            "total_loss": ...
        }
    """

    def __init__(self, cls_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        """
        outputs:
            {
                'output': logits [B, num_labels],
                'entropy_loss': optional,
                'load_balance_loss': optional
            }

        labels:
            {
                'label_ids': [B]
            }
        """

        logits = outputs["output"]       # [B, C]
        true_labels = labels["label_ids"].long()   # [B]

        # -------- 1. 分类 CrossEntropy --------
        loss_cls = self.ce_loss(logits, true_labels)
        total_loss = self.cls_weight * loss_cls

        # -------- 2. MoE entropy regularization --------
        if "entropy_loss" in outputs and outputs["entropy_loss"] is not None:
            total_loss += 0.1 * outputs["entropy_loss"]

        # -------- 3. MoE load balancing --------
        if "load_balance_loss" in outputs and outputs["load_balance_loss"] is not None:
            total_loss += 0.1 * outputs["load_balance_loss"]

        # -------- 4. 返回损失字典 --------
        return total_loss, {
            "loss_cls": loss_cls.item(),
            "total_loss": total_loss.item()
        }
