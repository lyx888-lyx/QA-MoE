"""Fast CPU selfchecks (no dataset download or BERT weight download)."""
import torch
from .data import corrupt_batch
from .metrics import compute
from .model import SparseExpertBank


def main():
    torch.manual_seed(5)
    label = torch.tensor([1.0, 0.5, -0.5, 0.0])
    sample = {"text":torch.tensor([[[101,2054,102,0],[1,1,1,0],[0,0,0,0]]]*4),
              "audio":torch.ones(4,4,5),
              "vision":torch.ones(4,4,20), "label":label}
    clean = corrupt_batch(sample, 0, 0, torch.ones(5), torch.ones(20))
    assert clean["missing"].sum() == 0
    assert torch.equal(sample["text"], clean["text"])
    blank = corrupt_batch(sample, 1, 1, torch.ones(5), torch.ones(20))
    assert blank["missing"].all()
    assert blank["audio"].abs().sum() == 0 and blank["vision"].abs().sum() == 0
    v = compute(label.numpy(),label.numpy())
    assert v["Acc_7"] == 100 and v["MAE"] == 0
    bank = SparseExpertBank(16,8,3)
    inp = torch.randn(2,3,16,requires_grad=True)
    fused,gates = bank(inp)
    assert fused.shape == inp.shape and gates.shape == (2,3,8)
    fused.square().mean().backward()
    assert inp.grad is not None and torch.isfinite(inp.grad).all()
    print("PASS: missing/noise, original labels and sparse expert gradients")


if __name__ == "__main__":
    main()
