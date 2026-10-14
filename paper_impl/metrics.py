"""MOSI/MOSEI metrics on original [-3,3] regression labels."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute(predictions, targets):
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    gold = np.asarray(targets, dtype=np.float64).reshape(-1)
    if pred.shape != gold.shape or not len(pred):
        raise ValueError("Predictions/targets missing or shape mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(gold).all():
        raise ValueError("Nonfinite prediction/label")
    seven = accuracy_score(np.rint(np.clip(gold, -3, 3)),
                           np.rint(np.clip(pred, -3, 3)))
    nz = gold != 0
    if nz.any():
        y, p = gold[nz] > 0, pred[nz] > 0
        acc2 = accuracy_score(y, p)
        f1 = f1_score(y, p, average="weighted", zero_division=0)
    else:
        acc2, f1 = float("nan"), float("nan")
    inclusive = accuracy_score(gold >= 0, pred >= 0)
    corr = np.corrcoef(pred, gold)[0, 1] if len(pred) > 1 and np.std(pred) > 0 and np.std(gold) > 0 else float("nan")
    return {
        "Acc_7": round(float(seven) * 100, 2),
        "Acc_2_nonzero": round(float(acc2) * 100, 2),
        "F1_nonzero": round(float(f1) * 100, 2),
        "Acc_2_including_zero": round(float(inclusive) * 100, 2),
        "MAE": round(float(np.mean(np.abs(pred - gold))), 4),
        "Corr": round(float(corr), 4),
    }
