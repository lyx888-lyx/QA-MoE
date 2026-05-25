import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from thop import profile, clever_format


def multiclass_acc(preds, truths):
    """计算多分类准确率（四舍五入后比较）"""
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def compute_mosi_metrics(y_pred, y_true):
    """
    Compute all MOSI/MOSEI metrics.

    y_pred, y_true: numpy arrays of floats.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    print(y_pred.shape, y_true.shape)
    # clip ranges for 7-class and 5-class eval
    preds_a7 = np.clip(y_pred, -3, 3)
    truth_a7 = np.clip(y_true, -3, 3)

    preds_a5 = np.clip(y_pred, -2, 2)
    truth_a5 = np.clip(y_true, -2, 2)

    # MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # Corr
    corr = np.corrcoef(y_pred, y_true)[0][1] if len(y_true) > 1 else 0.0

    # ACC
    acc_7 = multiclass_acc(preds_a7, truth_a7)
    acc_5 = multiclass_acc(preds_a5, truth_a5)

    # Binary accuracy and F1
    nonzero_idx = np.where(y_true != 0)[0]
    binary_truth = (y_true[nonzero_idx] > 0)
    binary_pred = (y_pred[nonzero_idx] > 0)

    acc_2 = accuracy_score(binary_truth, binary_pred)
    f1 = f1_score(binary_truth, binary_pred, average="weighted")

    return {
        "MAE": round(float(mae), 4),
        "Corr": round(float(corr), 4),
        "Acc_7": round(float(acc_7), 4),
        "Acc_5": round(float(acc_5), 4),
        "Acc_2": round(float(acc_2), 4),
        "F1": round(float(f1), 4),
    }

def compute_mintrec_metrics(y_pred, y_true, show_confusion=False):
    """
    Multi-class classification metrics for MIntRec.
    Accepts logits, probabilities, or class indices.
    """

    # Convert tensors → numpy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # --------- STEP 1: convert logits to class ids ---------
    if y_pred.ndim > 1:         # raw logits case (B, C)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        # continuous values → round? NO. classification must be int
        # ensure integer class
        y_pred = y_pred.astype(int)

    y_true = y_true.astype(int)

    # --------- STEP 2: compute metrics ---------
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")

    results = {
        "Acc":  round(float(acc), 4),
        "F1":   round(float(f1), 4),
        "Prec": round(float(prec), 4),
        "Rec":  round(float(rec), 4),
    }

    if show_confusion:
        results["Confusion"] = confusion_matrix(y_true, y_pred)

    return results


import time
import torch
import numpy as np
from tqdm import tqdm
try:
    from thop import profile, clever_format
except ImportError:
    print("Warning: 'thop' library not found. FLOPs calculation will be skipped. (pip install thop)")

@torch.no_grad()
def evaluate_one_epoch(
    config,
    model,
    dataloader,
    loss_fn,                   
    contrastive_loss_fn=None,
    device="cuda",
    use_contrastive=False,
    analyze_efficiency=True  # 🔥 新增开关，控制是否打印效率分析
):
    model.eval()

    # ====================================================
    # 🚀 PART 0: Computational Efficiency Analysis
    # ====================================================
    if analyze_efficiency:
        print("\n" + "="*40)
        print("   Starting Efficiency Analysis...")
        print("="*40)

        # 1. Parameter Count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f} M")

        # 2. Get a Sample Batch (用于 FLOPs 和 Latency 测试)
        # 注意：这里取一个 batch 不会影响后面正式的评估循环
        try:
            sample_batch = next(iter(dataloader))
            
            # 解析 Sample Batch (逻辑同下面循环)
            if config.dataset in ["MOSI", "MOSEI"]:
                s_text   = sample_batch["text"].to(device)
                s_audio  = sample_batch["audio"].to(device)
                s_vision = sample_batch["vision"].to(device)
            elif config.dataset == "MIntRec":
                s_text = sample_batch['text_feats'].to(device)
                s_audio = sample_batch['audio_feats'].to(device)
                s_vision = sample_batch['video_feats'].to(device)

            s_miss_t = sample_batch["missing_text"].to(device)
            s_miss_a = sample_batch["missing_audio"].to(device)
            s_miss_v = sample_batch["missing_vision"].to(device)
            
            # 构造输入元组 (必须与 model.forward 的参数顺序一致，或者适配 thop)
            # 这里的输入对应 model(text, audio, vision, ...)
            dummy_inputs = (s_text, s_audio, s_vision, s_miss_t, s_miss_a, s_miss_v)

            # 3. FLOPs Calculation (使用 thop)
            try:
                # profile 需要模型和输入，custom_ops 用于忽略不支持的算子警告
                flops, params = profile(model, inputs=dummy_inputs, verbose=False)
                flops_str, params_str = clever_format([flops, params], "%.3f")
                # thop 计算的是总 FLOPs，我们需要除以 batch size 得到 per sample
                batch_size_sample = s_text.shape[0]
                print(f"GFLOPs (Total Batch={batch_size_sample}): {flops_str}")
                # 手动估算 per sample (thop返回的是总数)
                print(f"GFLOPs (Per Sample approx): {flops / 1e9 / batch_size_sample:.4f} G")
            except Exception as e:
                print(f"FLOPs calculation failed: {e}")

            # 4. Inference Latency & Throughput
            # 预热 GPU
            print("Warming up for latency test...")
            for _ in range(10):
                _ = model(*dummy_inputs)
            
            # 正式测量
            iterations = 50
            start_time = time.time()
            for _ in range(iterations):
                _ = model(*dummy_inputs)
                if device == "cuda":
                    torch.cuda.synchronize() # 🔥 关键：等待 GPU 执行完毕
            end_time = time.time()

            total_time = end_time - start_time
            avg_latency = (total_time / iterations) * 1000 # ms
            # Throughput = (Batch Size * Iterations) / Total Time
            throughput = (s_text.shape[0] * iterations) / total_time

            print(f"Inference Latency (Batch={s_text.shape[0]}): {avg_latency:.2f} ms")
            print(f"Throughput: {throughput:.2f} samples/sec")

        except StopIteration:
            print("Dataloader is empty, skipping efficiency analysis.")
        
        print("="*40 + "\n")

    # ====================================================
    # 🚀 PART 1: Standard Evaluation Loop
    # ====================================================
    total_loss = 0.0
    total_main_loss = 0.0
    total_batches = 0

    preds, trues = [], []

    # 这里重新开始循环，不会受前面 next(iter(...)) 的影响
    for batch in tqdm(dataloader, desc="Evaluating"):

        # ------------------------------------------------
        # 1) Load batch
        # ------------------------------------------------
        if config.dataset in ["MOSI", "MOSEI"]:
            text   = batch["text"].to(device)
            audio  = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["regression_labels"].to(device)
            labels_dict = {"reg": labels}

        elif config.dataset == "MIntRec":
            text = batch['text_feats'].to(device)
            audio = batch['audio_feats'].to(device)
            vision = batch['video_feats'].to(device)
            labels = batch['label_ids'].to(device)
            labels_dict = {"label_ids": labels}

        missing_text   = batch["missing_text"].to(device)
        missing_audio  = batch["missing_audio"].to(device)
        missing_vision = batch["missing_vision"].to(device)

        # ------------------------------------------------
        # 2) Forward
        # ------------------------------------------------
        outputs = model(
            text,
            audio,
            vision,
            missing_text,
            missing_audio,
            missing_vision
        )
        fused_z = outputs["fused_z"]

        # ------------------------------------------------
        # 3) Optional contrastive
        # ------------------------------------------------
        if use_contrastive and contrastive_loss_fn is not None:
            contrastive_loss = contrastive_loss_fn(fused_z, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------
        # 4) Prepare loss dict
        # ------------------------------------------------
        outputs_for_loss = {
            "output": outputs["output"],            # logits or regression output
            "fused_z": fused_z,
            "contrastive_loss": contrastive_loss,
            "entropy_loss": outputs.get("entropy_loss", None),
            "load_balance_loss": None
        }

        # ------------------------------------------------
        # 5) Compute loss with unified API
        # ------------------------------------------------
        loss, loss_dict = loss_fn(outputs_for_loss, labels_dict)

        # ------------------------------------------------
        # 6) Add MoE load balancing loss (eval-friendly)
        # ------------------------------------------------
        if outputs.get("gating_weights", None) is not None:
            gating_weights = outputs["gating_weights"]
            importance = gating_weights.mean(dim=0)
            load_balance_loss = (importance * importance).sum()
            loss += config.load_balance_weight * load_balance_loss

            outputs_for_loss["load_balance_loss"] = load_balance_loss

        total_loss += loss.item()

        # regression vs classification logging
        if "loss_reg" in loss_dict:
            total_main_loss += loss_dict["loss_reg"]
        if "loss_cls" in loss_dict:
            total_main_loss += loss_dict["loss_cls"]

        total_batches += 1

        # ------------------------------------------------
        # 7) Store predictions for metric calculation
        # ------------------------------------------------
        if config.dataset in ["MOSI", "MOSEI"]:
            pred = outputs["output"].squeeze(-1)   # [B]
        else:
            pred = outputs["output"].argmax(dim=-1)  # [B]

        preds.extend(pred.cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())

    # Convert list → numpy
    preds = np.array(preds)
    trues = np.array(trues)

    avg_loss = total_loss / max(total_batches, 1)
    avg_main_loss = total_main_loss / max(total_batches, 1)

    # ------------------------------------------------
    # 8) Compute metrics
    # ------------------------------------------------
    if config.dataset in ["MOSI", "MOSEI"]:
        # 假设 compute_mosi_metrics 已经在外部定义
        metrics = compute_mosi_metrics(preds, trues)
        metrics["val_reg_loss"] = avg_main_loss

    else:  # MIntRec
        # 假设 compute_mintrec_metrics 已经在外部定义
        metrics = compute_mintrec_metrics(preds, trues)

    metrics["val_loss"] = avg_loss

    return metrics