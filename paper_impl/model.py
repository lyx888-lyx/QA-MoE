"""Paper-oriented QA-MoE (Eqs. 3–10), not a claim of author-identical code."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class SequentialEncoder(nn.Module):
    def __init__(self, input_dim, hidden, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Ignore padded/zero input rows when averaging. Nonzero is not a label.
        valid = x.abs().sum(dim=-1) > 0
        hidden, _ = self.rnn(self.proj(x))
        hidden = self.dropout(self.output(hidden))
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1)
        return (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom


class GLUExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        return self.value(x) * torch.sigmoid(self.gate(x))


class SparseExpertBank(nn.Module):
    """Actually evaluates only the chosen top-k experts for each modality/sample."""
    def __init__(self, dim, experts=8, top_k=3):
        super().__init__()
        if not (1 <= top_k <= experts):
            raise ValueError("1 <= top_k <= experts required")
        self.experts = nn.ModuleList(GLUExpert(dim) for _ in range(experts))
        self.router = nn.Linear(dim, experts)
        self.top_k = top_k

    def forward(self, mu):
        # [B,3,D], one semantic router decision per modality
        batch, modalities, dim = mu.shape
        flat = mu.reshape(batch * modalities, dim)
        probabilities = torch.softmax(self.router(flat), dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        selected = flat.new_zeros((flat.shape[0], self.top_k, dim))
        for expert_id, expert in enumerate(self.experts):
            row, slot = (indices == expert_id).nonzero(as_tuple=True)
            if row.numel():
                selected.index_put_((row, slot), expert(flat[row]))
        values = (selected * weights.unsqueeze(-1)).sum(dim=1)
        return values.reshape(batch, modalities, dim), probabilities.reshape(batch, modalities, -1)


class PaperQAMoE(nn.Module):
    """3 Gaussian modality embeddings -> quality-weighted sparse MoE -> fallback -> dual head.

    Equations constrained by paper; some unspecified encoder/fusion details are engineering
    choices and may differ from the authors' unreleased reference implementation.
    """
    def __init__(self, bert_path="bert-base-uncased", audio_dim=74, vision_dim=35,
                 hidden=128, experts=8, top_k=3, dropout=0.1, logvar_floor=None):
        super().__init__()
        self.logvar_floor = logvar_floor
        self.bert = BertModel.from_pretrained(bert_path)
        self.text_projection = nn.Linear(self.bert.config.hidden_size, hidden)
        self.audio = SequentialEncoder(audio_dim, hidden, dropout)
        self.vision = SequentialEncoder(vision_dim, hidden, dropout)
        self.mean_heads = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
        self.var_heads = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
        self.bank = SparseExpertBank(hidden, experts, top_k)
        self.global_prior = nn.Parameter(torch.zeros(hidden))
        self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden),
                                  nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.pred_head = nn.Linear(hidden, 1)
        self.logvar_head = nn.Linear(hidden, 1)

    def forward(self, text, audio, vision, missing=None):
        batch = text.shape[0]
        if missing is None:
            missing = torch.zeros((batch, 3), device=text.device, dtype=torch.bool)
        input_ids, attention, types = text[:, 0].long(), text[:, 1].long(), text[:, 2].long()
        # For physically missing text, feed a valid minimal token sequence to BERT,
        # then explicitly zero its pooled representation.
        missing_text = missing[:, 0]
        if missing_text.any():
            input_ids = input_ids.clone()
            attention = attention.clone()
            types = types.clone()
            input_ids[missing_text] = 0
            attention[missing_text] = 0
            input_ids[missing_text, 0] = 101
            attention[missing_text, 0] = 1
            types[missing_text] = 0
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention,
                             token_type_ids=types).last_hidden_state
        pooled_text = ((bert_out * attention.unsqueeze(-1)).sum(dim=1) /
                       attention.sum(dim=1, keepdim=True).clamp(min=1))
        txt = self.text_projection(pooled_text)
        embeddings = torch.stack((txt, self.audio(audio), self.vision(vision)), dim=1)
        embeddings = embeddings.masked_fill(missing.unsqueeze(-1), 0)
        mu = torch.stack([head(embeddings[:, i]) for i, head in enumerate(self.mean_heads)], 1)
        var = torch.stack([F.softplus(head(embeddings[:, i])) + 1e-6
                           for i, head in enumerate(self.var_heads)], 1)
        # Equation (6): inverse expected predicted variance
        reliability = 1.0 / (1.0 + var.mean(dim=-1))
        reliability = reliability.masked_fill(missing, 0.0)
        per_modality, gating = self.bank(mu)
        # Equation (7): learned global prior interpolation
        refined = (reliability.unsqueeze(-1) * per_modality
                   + (1 - reliability).unsqueeze(-1) * self.global_prior)
        h = self.fuse(refined.reshape(batch, -1))
        raw_log_var = self.logvar_head(h).squeeze(-1)
        if self.logvar_floor is None:
            log_var = raw_log_var.clamp(-10.0, 10.0)
        else:
            # Smooth lower bound: unlike clamp, gradients survive below the floor.
            log_var = (self.logvar_floor +
                       F.softplus(raw_log_var - self.logvar_floor)).clamp(max=10.0)
        return {"pred": self.pred_head(h).squeeze(-1),
                "log_var": log_var,
                "reliability": reliability, "mu": mu, "variance": var, "gates": gating}

    @staticmethod
    def nll(output, target):
        target = target.reshape(-1)
        predicted = output["pred"].reshape(-1)
        log_var = output["log_var"].reshape(-1)
        if predicted.shape != target.shape:
            raise ValueError(f"Prediction {predicted.shape} vs label {target.shape}")
        return (0.5 * (torch.exp(-log_var) * (predicted - target).square() + log_var)).mean()
