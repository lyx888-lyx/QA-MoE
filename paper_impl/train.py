"""Paper-oriented QA-MoE: single-GPU clean/spectrum train and evaluation."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from .data import AlignedSentimentDataset, corrupt_batch, resolve_data
from .metrics import compute
from .model import PaperQAMoE


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    p.add_argument("--data", required=True, help="Aligned .pkl file, or directory")
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--protocol", choices=["clean", "spectrum"], default="clean")
    p.add_argument("--bert-path", default="bert-base-uncased")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output", default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bert-lr", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--logvar-floor", type=float, default=None,
                   help="Experimental: smooth log variance lower bound; e.g. -1.0")
    p.add_argument("--aux-huber-weight", type=float, default=0.0,
                   help="Experimental Huber auxiliary regression loss")
    p.add_argument("--router-balance-weight", type=float, default=0.0,
                   help="Experimental soft router batch-balance penalty")
    p.add_argument("--skip-test-after-training", action="store_true",
                   help="Keep final test set untouched while tuning on validation")
    p.add_argument("--freeze-bert", action="store_true",
                   help="Resource-saving deviation from the paper")
    p.add_argument("--max-steps", type=int, default=0,
                   help="DEBUG ONLY: train at most N batches per epoch")
    p.add_argument("--eval-grid", action="store_true")
    p.add_argument("--grid-step", type=float, default=0.1)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def get_inputs(batch):
    return (batch["text"], batch["audio"], batch["vision"], batch.get("missing"))


def corrupt(batch, mode, ds, rng=None, lam=None, eta=None):
    if lam is None:
        if mode == "spectrum":
            # Independent lambda and eta, sampled once per training mini-batch
            lam, eta = random.random(), random.random()
        else:
            lam, eta = 0.0, 0.0
    return corrupt_batch(batch, lam, eta, ds.ref_a_tensor, ds.ref_v_tensor, rng)


def evaluate(model, loader, ds, device, amp, seed, lam=0.0, eta=0.0):
    model.eval()
    pred, target, losses = [], [], []
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for batch in loader:
            sample = corrupt(move(batch, device), "clean", ds, rng, lam, eta)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(*get_inputs(sample))
                loss = model.nll(output, sample["label"])
            losses.append(loss.item())
            pred.extend(output["pred"].float().cpu().numpy().reshape(-1).tolist())
            target.extend(sample["label"].cpu().numpy().reshape(-1).tolist())
    result = compute(pred, target)
    result["NLL"] = round(float(np.mean(losses)), 5)
    return result


def main():
    args = arguments()
    if args.batch_size < 1 or args.epochs < 1:
        raise ValueError("batch-size and epochs must be positive")
    if args.aux_huber_weight < 0 or args.router_balance_weight < 0:
        raise ValueError("Experimental loss weights must be nonnegative")
    if args.logvar_floor is not None and not (-10.0 < args.logvar_floor <= 3.0):
        raise ValueError("logvar-floor must be between -10 and 3")
    if args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("--mode eval requires --checkpoint")
        metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        saved_options = metadata.get("args", {})
        saved_floor = saved_options.get("logvar_floor")
        if args.logvar_floor is not None and args.logvar_floor != saved_floor:
            raise ValueError("Requested --logvar-floor differs from the saved checkpoint")
        args.logvar_floor = saved_floor
    seed_everything(args.seed)
    if args.max_steps:
        print(f"WARNING: --max-steps={args.max_steps} truncates every training epoch; results are NOT comparable with the paper.")
    print("TRAIN FLAGS:", json.dumps({k:v for k,v in vars(args).items() if k not in ("bert_path", "data")}, ensure_ascii=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    amp = args.amp and device.type == "cuda"
    path = resolve_data(args.data)
    output_dir = Path(args.output or f"runs/{args.dataset}_{args.protocol}")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {k: AlignedSentimentDataset(path, k) for k in ("train", "valid", "test")}
    for ds in datasets.values():
        ds.ref_a_tensor = torch.from_numpy(ds.ref_a.copy()).float()
        ds.ref_v_tensor = torch.from_numpy(ds.ref_v.copy()).float()
    loaders = {k: DataLoader(ds, batch_size=args.batch_size, shuffle=(k == "train"),
                             num_workers=args.workers) for k, ds in datasets.items()}
    print(f"Data: {path}; splits={[(k,len(v)) for k,v in datasets.items()]}")
    ds = datasets["train"]
    print(f"Audio dim={ds.audio.shape[-1]}, vision dim={ds.vision.shape[-1]}, original label range={ds.labels.min():.2f}..{ds.labels.max():.2f}")
    model = PaperQAMoE(args.bert_path, ds.audio.shape[-1], ds.vision.shape[-1],
                       args.hidden, args.experts, args.top_k,
                       logvar_floor=args.logvar_floor).to(device)
    if args.freeze_bert:
        for p in model.bert.parameters():
            p.requires_grad_(False)
        print("WARNING: frozen BERT is not strict end-to-end reproduction")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M; device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device))
    if args.mode == "train":
        other = [p for n, p in model.named_parameters() if not n.startswith("bert.") and p.requires_grad]
        bert = [p for p in model.bert.parameters() if p.requires_grad]
        groups = [{"params": other, "lr": args.lr}]
        if bert:
            groups.append({"params": bert, "lr": args.bert_lr})
        optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        best = float("inf")
        with (output_dir / "history.jsonl").open("w", encoding="utf-8") as history:
            for epoch in range(args.epochs):
                model.train()
                start = time.monotonic()
                total = 0.0
                nll_total = 0.0
                huber_total = 0.0
                balance_total = 0.0
                steps = 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                for raw in loaders["train"]:
                    sample = corrupt(move(raw, device), args.protocol, ds)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp):
                        output = model(*get_inputs(sample))
                        loss_nll = model.nll(output, sample["label"])
                        loss_huber = F.smooth_l1_loss(output["pred"], sample["label"], beta=0.5)
                        soft_usage = output["gates"].mean(dim=(0, 1))
                        num_experts = soft_usage.numel()
                        loss_balance = num_experts * soft_usage.square().sum() - 1.0
                        loss = (loss_nll +
                                args.aux_huber_weight * loss_huber +
                                args.router_balance_weight * loss_balance)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss epoch={epoch+1}, step={steps+1}")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total += loss.item()
                    nll_total += loss_nll.item()
                    huber_total += loss_huber.item()
                    balance_total += loss_balance.item()
                    steps += 1
                    if args.max_steps and steps >= args.max_steps:
                        break
                scheduler.step()
                validation = evaluate(model, loaders["valid"], datasets["valid"], device, amp, args.seed)
                peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device.type == "cuda" else None
                log = {"epoch":epoch+1, "train_NLL":round(nll_total/max(steps,1),5),
                       "train_objective":round(total/max(steps,1),5),
                       "train_Huber":round(huber_total/max(steps,1),5),
                       "train_router_penalty":round(balance_total/max(steps,1),5),
                       "validation":validation, "peak_gpu_GiB":peak,
                       "seconds":round(time.monotonic()-start,1), "train_steps":steps,
                       "expected_steps":len(loaders["train"]), "max_steps":args.max_steps,
                       "logvar_floor":args.logvar_floor, "aux_huber_weight":args.aux_huber_weight,
                       "router_balance_weight":args.router_balance_weight}
                history.write(json.dumps(log, ensure_ascii=False) + "\n")
                history.flush()
                print(json.dumps(log, ensure_ascii=False))
                if validation["MAE"] < best:
                    best = validation["MAE"]
                    torch.save({"state_dict":model.state_dict(), "args":vars(args),
                                "audio_dim": ds.audio.shape[-1], "vision_dim":ds.vision.shape[-1],
                                "epoch":epoch+1, "val_MAE":best}, output_dir/"best.pt")
        ckpt_path = output_dir/"best.pt"
        if args.skip_test_after_training:
            print("Validation-only experiment: final test set NOT evaluated.")
            print(f"Best checkpoint: {ckpt_path}")
            return
    else:
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint path/to/best.pt for --mode eval")
        ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    original = ckpt.get("args", {})
    for name in ("hidden", "experts", "top_k"):
        if getattr(args, name) != original.get(name, getattr(args, name)):
            raise ValueError(f"Mismatch in {name} with checkpoint. Use the checkpoint settings.")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    result = evaluate(model, loaders["test"], datasets["test"], device, amp, args.seed)
    print("CLEAN TEST:", json.dumps(result))
    (output_dir/"test_clean.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.eval_grid:
        if not 0 < args.grid_step <= 0.7:
            raise ValueError("grid-step must be (0,0.7]")
        grid = np.round(np.arange(0, 0.70001, args.grid_step), 5)
        with (output_dir/"spectrum_grid.csv").open("w", newline="") as f:
            keys = ["lambda","eta","Acc_7","Acc_2_nonzero","F1_nonzero",
                    "Acc_2_including_zero","MAE","Corr","NLL"]
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for eta in grid:
                for lam in grid:
                    r = evaluate(model, loaders["test"], datasets["test"], device,
                                 amp, args.seed, float(lam), float(eta))
                    writer.writerow({"lambda":float(lam), "eta":float(eta), **r})
                    print(f"lambda={lam:.2f}, eta={eta:.2f}, Acc_7={r['Acc_7']}")
        print("Grid:", output_dir/"spectrum_grid.csv")


if __name__ == "__main__":
    main()
