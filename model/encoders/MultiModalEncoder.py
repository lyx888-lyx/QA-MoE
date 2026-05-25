import torch
import torch.nn as nn
from transformers import BertModel

# ============================================================
# Vision Encoder (BiLSTM 增强版)
# ============================================================
class VisionEncoder(nn.Module):
    """
    Input:  [B, T, D_in]
    Output: [B, T, 128]
    Refactored to BiLSTM for better stability and time-series modeling
    """
    def __init__(self, input_dim, output_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(VisionEncoder, self).__init__()

        # 1. 预处理投影层 (把原始特征映射到 hidden space)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 核心：双向 LSTM (比 Transformer 更适合未预训练的连续信号)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim, # 双向拼接后会变成 2*hidden，后面再降维
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 3. 输出投影 (把 2*hidden 降回 output_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, missing_vision=None):
        """
        x: [B, T, D]
        missing_vision: [B] bool
        """
        B, T, _ = x.shape

        # 安全清洗
        x = torch.nan_to_num(x)

        # --- Missing Case: 强制物理阻断 ---
        if missing_vision is not None and missing_vision.any():
            # 初始化全 0 输出 (保证无噪声)
            out = torch.zeros(B, T, 128, device=x.device)
            valid = ~missing_vision # [B]

            if valid.any():
                # 只计算有效的样本
                x_valid = x[valid] # [N_valid, T, D]
                
                # Forward
                feat = self.proj(x_valid)
                self.lstm.flatten_parameters() # 内存优化
                lstm_out, _ = self.lstm(feat)
                out_valid = self.out_proj(lstm_out)
                
                # 填回
                out[valid] = out_valid

            return out

        # --- Normal Case ---
        feat = self.proj(x)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(feat)
        return self.out_proj(lstm_out)


# ============================================================
# Text Encoder (清洗版)
# ============================================================
class TextEncoder(nn.Module):
    """
    text: [B, 3, 50]
    -> input_ids, attention_mask, token_type_ids
    """
    def __init__(self, pretrained='bert-base-uncased', finetune=True):
        super().__init__()
        self.finetune = finetune
        self.bert = BertModel.from_pretrained(pretrained)
        self.out_proj = nn.Linear(768, 128)
        # 移除了 dangling parameter (missing_embedding)，因为它没被用到

    def forward(self, text, missing_text=None):
        # 🔥 1. 强制清洗 NaNs
        if torch.isnan(text).any():
            text = torch.nan_to_num(text, nan=0.0)

        # 🔥 2. 转为 Long
        input_ids = text[:, 0, :].long()
        attention_mask = text[:, 1, :].long()
        token_type_ids = text[:, 2, :].long()

        # 🔥 3. 安全检查 (Debug用)
        vocab_size = self.bert.config.vocab_size
        if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
            # 强制修正
            input_ids = torch.clamp(input_ids, min=0, max=vocab_size-1)

        B, seq_len = input_ids.shape

        # --- Missing Case ---
        if missing_text is not None and missing_text.any():
            out = torch.zeros(B, seq_len, 128, device=text.device)
            valid = ~missing_text

            if valid.any():
                outputs = self.bert(
                    input_ids=input_ids[valid],
                    attention_mask=attention_mask[valid],
                    token_type_ids=token_type_ids[valid]
                )
                out[valid] = self.out_proj(outputs.last_hidden_state)
            
            return out

        # --- Normal Case ---
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return self.out_proj(outputs.last_hidden_state)


# ============================================================
# Audio Encoder (BiLSTM 增强版)
# ============================================================
class AudioEncoder(nn.Module):
    """
    Input: [B,T,D]
    Output: [B,T,128]
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(AudioEncoder, self).__init__()
        
        # 1. 增加预处理层 (对齐 VisionEncoder 结构)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 增加深度 (num_layers=2) 以对抗 BERT 的统治力
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim, # 双向拼接后 2*hidden
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 3. 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x, missing_audio=None):
        B, T, _ = x.shape
        
        # 安全清洗
        x = torch.nan_to_num(x)

        # --- Missing Case ---
        if missing_audio is not None and missing_audio.any():
            out = torch.zeros(B, T, 128, device=x.device)
            valid = ~missing_audio

            if valid.any():
                feat = self.proj(x[valid])
                self.bilstm.flatten_parameters()
                lstm_out, _ = self.bilstm(feat)
                out[valid] = self.out_proj(lstm_out)

            return out

        # --- Normal Case ---
        feat = self.proj(x)
        self.bilstm.flatten_parameters()
        lstm_out, _ = self.bilstm(feat)
        return self.out_proj(lstm_out)