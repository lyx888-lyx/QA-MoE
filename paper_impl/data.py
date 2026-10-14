"""MOSI/MOSEI aligned features; original [-3, 3] labels and controlled corruption."""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_data(path):
    path = Path(path).expanduser()
    if path.is_file():
        if "unaligned" in path.name.lower():
            raise ValueError("This entrypoint reproduces ALIGNED experiments only; use aligned_50.pkl.")
        return path
    for name in ("aligned_50.pkl", "aligned_50-001.pkl"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    matches = sorted(path.glob("aligned_50*.pkl")) if path.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"No unique aligned_50*.pkl found under {path}. Pass the exact file.")


class AlignedSentimentDataset(Dataset):
    """Loads one split. Raw sentiment labels are NEVER Z-score normalized."""
    def __init__(self, data, split):
        path = resolve_data(data)
        with path.open("rb") as f:
            all_data = pickle.load(f)
        if split not in all_data:
            raise KeyError(f"Split {split} missing. Available: {list(all_data)}")
        self.splits = all_data
        self.split = split
        src = all_data[split]
        for key in ("text_bert", "audio", "vision", "regression_labels"):
            if key not in src:
                raise KeyError(f"Missing key {key} in {split}; expected MMSA aligned_50.pkl")
        self.text = np.asarray(src["text_bert"])
        self.audio = np.asarray(src["audio"], dtype=np.float32)
        self.vision = np.asarray(src["vision"], dtype=np.float32)
        self.labels = np.asarray(src["regression_labels"], dtype=np.float32).reshape(-1)
        if self.text.ndim != 3 or self.text.shape[1] != 3:
            raise ValueError(f"text_bert must be [N,3,T] BERT token triples; got {self.text.shape}")
        if self.audio.ndim != 3 or self.vision.ndim != 3:
            raise ValueError("Audio and vision must be [N,T,D]")
        sizes = [len(self.text), len(self.audio), len(self.vision), len(self.labels)]
        if len(set(sizes)) != 1:
            raise ValueError(f"Inconsistent split lengths: {sizes}")
        if not np.isfinite(self.labels).all():
            raise ValueError("Nonfinite sentiment labels")
        if self.labels.min() < -3.01 or self.labels.max() > 3.01:
            raise ValueError("Expected original MOSI/MOSEI scores in [-3,3], not normalized labels")
        train = all_data["train"]
        self.ref_a = np.asarray(train["audio"], dtype=np.float32).std(axis=(0, 1)).clip(1e-6)
        self.ref_v = np.asarray(train["vision"], dtype=np.float32).std(axis=(0, 1)).clip(1e-6)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "text": torch.from_numpy(self.text[index].astype(np.int64)),
            "audio": torch.from_numpy(self.audio[index].copy()),
            "vision": torch.from_numpy(self.vision[index].copy()),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def corrupt_batch(batch, intensity, missing_rate, ref_a, ref_v, rng=None):
    """Paper Appendix A.1: λ and η shared within a batch, independent masks per example/modality.

    Text corruption uses BERT [MASK]=103 for valid tokens (except CLS/SEP).
    Audio/vision use AWGN scaled by the *training-set* feature std.
    """
    if not 0 <= intensity <= 1 or not 0 <= missing_rate <= 1:
        raise ValueError("lambda and eta must both be in [0,1]")
    x = {k: v.clone() for k, v in batch.items()}
    text, audio, vision = x["text"], x["audio"], x["vision"]
    device = audio.device
    b = audio.shape[0]
    rand = lambda shape: torch.rand(shape, device=device, generator=rng)
    if intensity:
        valid_tokens = (text[:, 1, :] > 0) & (text[:, 0, :] != 101) & (text[:, 0, :] != 102)
        token_drop = (rand(valid_tokens.shape) < intensity) & valid_tokens
        text[:, 0, :][token_drop] = 103
        audio.add_(torch.randn(audio.shape, device=device, generator=rng) *
                   intensity * ref_a.to(device).view(1, 1, -1))
        vision.add_(torch.randn(vision.shape, device=device, generator=rng) *
                    intensity * ref_v.to(device).view(1, 1, -1))
    missing = rand((b, 3)) < missing_rate
    text[missing[:, 0]] = 0
    audio[missing[:, 1]] = 0
    vision[missing[:, 2]] = 0
    x["missing"] = missing
    return x
