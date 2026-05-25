# model/moe/ExpertBank.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUExpert(nn.Module):
    """
    GLU Expert:
    y = (W1 x) * σ(W2 x)
    input_dim == output_dim == expert_dim
    """
    def __init__(self, expert_dim):
        super().__init__()
        self.fc1 = nn.Linear(expert_dim, expert_dim)
        self.fc2 = nn.Linear(expert_dim, expert_dim)

        # 初始化（更稳定）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # GLU: gate-controlled transformation
        return self.fc1(x) * torch.sigmoid(self.fc2(x))


class ExpertBank(nn.Module):
    """
    Holds N experts. Each is a GLUExpert.
    Called via expert_bank(expert_id, x)
    """
    def __init__(self, num_experts, expert_dim):
        super().__init__()

        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            GLUExpert(expert_dim) for _ in range(num_experts)
        ])

    def forward(self, expert_id, x):
        """
        expert_id: int
        x: [B, expert_dim]
        """
        return self.experts[expert_id](x)

    def get_total_expert_num(self):
        return self.num_experts
