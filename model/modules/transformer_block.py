import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, num_layers=1,
                 dropout=0.1, use_pooling=True):
        super(TransformerExpert, self).__init__()

        self.use_pooling = use_pooling

        # 输入投影到 Transformer 隐层维度
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer 层
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x:
            [B, T, D]  或
            [B, D]（非序列） → 自动补成 [B, 1, D]
        return:
            use_pooling=True → [B, H]
            else → [B, T, H]
        """

        # ---- (1) 如果是 [B, D]，自动变成 [B, 1, D] ----
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # ---- (2) 进入 Transformer ----
        x = self.input_proj(x)     # [B, T, D] -> [B, T, H]
        x = self.transformer(x)    # [B, T, H]
        x = self.norm(x)

        # ---- (3) 输出 pooled 或 seq ----
        if self.use_pooling:
            return x.mean(dim=1)   # [B, H]
        else:
            return x
