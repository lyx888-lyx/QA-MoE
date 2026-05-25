import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def compute_task_loss(pred, target, task_type: str = 'binary') -> torch.Tensor:
    """
    计算任务损失，根据任务类型选择合适的损失函数。
    支持 'binary', 'regression', 'quadrant' 三种任务类型。
    """
    if task_type == 'binary':
        return F.binary_cross_entropy_with_logits(pred, target.float())
    elif task_type == 'regression':
        return F.mse_loss(pred, target.float())
    elif task_type == 'quadrant':
        return F.cross_entropy(pred, target.long())
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

def compute_load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """
    计算负载平衡损失，使用 KL 散度衡量专家负载的均匀性。
    gate_weights: [B, K]，每个样本对每个专家的权重。
    """
    expert_load = gate_weights.mean(dim=0)  # [K]
    uniform = torch.full_like(expert_load, 1.0 / expert_load.size(0))  # 均匀分布
    return F.kl_div(expert_load.log(), uniform, reduction='batchmean')

def compute_metrics(pred, target, task_type: str = 'binary') -> dict:
    """
    计算任务相关的评估指标。
    支持 'binary', 'regression', 'quadrant' 三种任务类型。
    返回一个字典，包含各类指标。
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if task_type == 'binary':
        pred_label = (pred_np > 0.5).astype(int)
        pred = torch.sigmoid(pred)
        return {
            'accuracy': accuracy_score(target_np, pred_label),
            'f1_score': f1_score(target_np, pred_label, average='macro'),
        }
    elif task_type == 'regression':
        return {
            'mse': mean_squared_error(target_np, pred_np)
        }
    elif task_type == 'quadrant':
        return {
            'accuracy': accuracy_score(target_np, pred_np),
            'f1_score': f1_score(target_np, pred_np, average='macro'),
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


class NTXentLoss(nn.Module):
    """
    NT-Xent 对比损失函数（适用于多模态对齐，如 face ↔ EEG）
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        device = z1.device

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_matrix.fill_diagonal_(-1e9)

        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B)
        ], dim=0).to(device)

        return F.cross_entropy(sim_matrix, labels)


def multimodal_contrastive(z_dict, mask, loss_fn):
    """
    多模态对比损失调度器（支持缺失模态屏蔽）
    参数:
        z_dict: dict of modality → [B, D]
        mask:   [B, M] → 每个样本模态是否存在
        loss_fn: 通常为 NTXentLoss 实例
    返回:
        所有模态对的平均对比损失
    """
    total_loss = 0.0
    num_pairs = 0
    keys = list(z_dict.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            mi, mj = keys[i], keys[j]
            mask_i, mask_j = mask[:, i], mask[:, j]
            valid = (mask_i * mask_j).bool()
            if valid.sum() < 2:
                continue
            zi = z_dict[mi][valid]
            zj = z_dict[mj][valid]
            total_loss += loss_fn(zi, zj)
            num_pairs += 1

    return total_loss / max(num_pairs, 1)


z_face = torch.randn(32, 128)  # [B, D]
z_eeg = torch.randn(32, 128)   # [B, D]

z_dict = {"face": z_face, "eeg": z_eeg}  # [B, 128]
mask = torch.randn(32, 2)             # [B, 2]
loss_fn = NTXentLoss(temperature=0.07)
loss = multimodal_contrastive(z_dict, mask, loss_fn)
print(f"Contrastive Loss: {loss.item()}")
