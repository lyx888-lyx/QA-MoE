"""Diagnose an existing paper_impl checkpoint WITHOUT changing weights.

Read back recorded training flags, evaluate prediction uncertainty on validation data,
and report NLL/MAE calibration, modality reliability and expert routing.
Run from repository root: python -m paper_impl.diagnose --checkpoint runs/mosi_clean/best.pt --data PATH
"""
import argparse
import json
import math
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AlignedSentimentDataset
from .metrics import compute
from .model import PaperQAMoE


def summarize(values):
    z = np.asarray(values, dtype=np.float64).reshape(-1)
    if not z.size:
        return {}
    return {key: round(float(value), 5) for key, value in {
        "mean":z.mean(), "std":z.std(), "min":z.min(), "p05":np.quantile(z, 0.05),
        "median":np.median(z), "p95":np.quantile(z, 0.95), "max":z.max()
    }.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="Original aligned_50.pkl")
    ap.add_argument("--split", choices=["valid","test","train"], default="valid")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bert-path", default="", help="Override model location from the checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    opts = saved.get("args", {})
    print("CHECKPOINT:", json.dumps({
        "epoch":saved.get("epoch"), "best_val_MAE":saved.get("val_MAE"),
        "dataset":opts.get("dataset"), "protocol":opts.get("protocol"),
        "batch_size":opts.get("batch_size"), "max_steps":opts.get("max_steps"),
        "freeze_bert":opts.get("freeze_bert"), "amp":opts.get("amp"),
        "epochs":opts.get("epochs"), "seed":opts.get("seed"),
        "lr":opts.get("lr"), "bert_lr":opts.get("bert_lr")
    },ensure_ascii=False,indent=2))
    if opts.get("max_steps",0):
        print("CRITICAL WARNING: checkpoint used --max-steps > 0; this was a TRUNCATED training run, not comparable to the paper.")
    ds = AlignedSentimentDataset(args.data, args.split)
    loader = DataLoader(ds,batch_size=args.batch_size,shuffle=False)
    bert_path = args.bert_path or opts.get("bert_path","bert-base-uncased")
    model = PaperQAMoE(bert_path, ds.audio.shape[-1], ds.vision.shape[-1],
                       opts.get("hidden",128), opts.get("experts",8),opts.get("top_k",3)).to(device)
    model.load_state_dict(saved["state_dict"],strict=True)
    model.eval()
    pred, gold, logvar, r, gating = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= args.max_batches:
                break
            text,audio,vision = (batch[x].to(device) for x in ("text","audio","vision"))
            out = model(text,audio,vision)
            pred.append(out["pred"].float().cpu().numpy().reshape(-1))
            gold.append(batch["label"].numpy().reshape(-1))
            logvar.append(out["log_var"].float().cpu().numpy().reshape(-1))
            r.append(out["reliability"].float().cpu().numpy())
            gating.append(out["gates"].float().cpu().numpy())
    pred,gold,logvar=np.concatenate(pred),np.concatenate(gold),np.concatenate(logvar)
    rel=np.concatenate(r)
    gate=np.concatenate(gating)
    abs_err=np.abs(pred-gold)
    sigma=np.sqrt(np.exp(np.clip(logvar,-30,30)))
    nll=0.5*(np.exp(-logvar)*(pred-gold)**2+logvar)
    topk=np.argsort(gate,axis=-1)[...,-opts.get("top_k",3):].reshape(-1)
    count=Counter(int(x) for x in topk)
    report={
        "split":args.split,"N":int(pred.size),
        "metrics":compute(pred,gold),"NLL_mean":round(float(nll.mean()),4),
        "prediction":summarize(pred), "target":summarize(gold),
        "abs_error":summarize(abs_err),
        "log_variance":summarize(logvar),
        "predicted_sigma":summarize(sigma),
        "fraction_logvar_below_minus_5":round(float(np.mean(logvar < -5)),4),
        "empirical_one_sigma_coverage":round(float(np.mean(abs_err <= sigma)),4),
        "empirical_two_sigma_coverage":round(float(np.mean(abs_err <= 2*sigma)),4),
        "fraction_error_over_one":round(float(np.mean(abs_err > 1)),4),
        "reliability_mean_by_modality_text_audio_visual":[round(float(x),4) for x in rel.mean(axis=0)],
        "reliability_std_by_modality_text_audio_visual":[round(float(x),4) for x in rel.std(axis=0)],
        "topk_expert_select_counts":{str(k):count.get(k,0) for k in range(opts.get("experts",8))}
    }
    print("DIAGNOSTICS:",json.dumps(report,ensure_ascii=False,indent=2))
    if np.mean(abs_err <= sigma) < 0.40:
        print("WARNING: 1-sigma coverage is far below the ~0.68 normal reference: likely under-estimated prediction variance / miscalibration.")
    print("NOTE: calibration benchmarks assume Gaussian model and are diagnostic, not proof of a bug.")


if __name__ == "__main__":
    main()
