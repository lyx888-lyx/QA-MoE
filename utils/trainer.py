import torch
from tqdm import tqdm
import torch.nn as nn


def train_one_epoch(
    config,
    epoch,
    model,
    dataloader,
    optimizer,
    loss_fn,                   # RegressionMoELoss or ClassificationMoELoss
    contrastive_loss_fn=None,
    scheduler=None,
    device="cuda",
    use_contrastive=False
):

    model.train()

    total_loss = 0.0
    total_main_loss = 0.0
    total_batches = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):

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
        # 2) Forward pass
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
        # 3) Contrastive loss (optional)
        # ------------------------------------------------
        if use_contrastive and contrastive_loss_fn is not None:
            contrastive_loss = contrastive_loss_fn(fused_z, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # ------------------------------------------------
        # 4) Structured loss_dict for MoELoss
        # ------------------------------------------------
        outputs_for_loss = {
            "output": outputs["output"],                 # logits or regression
            "fused_z": fused_z,
            "contrastive_loss": contrastive_loss,
            "entropy_loss": outputs.get("entropy_loss", None),
            "load_balance_loss": None
        }

        # ------------------------------------------------
        # 5) Compute Primary Loss
        # ------------------------------------------------
        loss, main_loss_dict = loss_fn(outputs_for_loss, labels_dict)

        # ------------------------------------------------
        # 6) Expert Load Balancing
        # ------------------------------------------------
        if outputs.get("gating_weights", None) is not None:
            gating_weights = outputs["gating_weights"]     # [B, N]
            importance = gating_weights.mean(dim=0)        # [N]

            load_balance_loss = (importance * importance).sum()
            loss += config.load_balance_weight * load_balance_loss
            outputs_for_loss["load_balance_loss"] = load_balance_loss

        # ------------------------------------------------
        # 7) Backprop
        # ------------------------------------------------
        optimizer.zero_grad()
        loss.backward()

        # NaN check
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print("❌ NaN gradient in:", name)
                raise SystemExit

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ------------------------------------------------
        # 8) Logging
        # ------------------------------------------------
        total_loss += loss.item()

        # regression vs classification logging
        if "loss_reg" in main_loss_dict:
            total_main_loss += main_loss_dict["loss_reg"]
        elif "loss_cls" in main_loss_dict:
            total_main_loss += main_loss_dict["loss_cls"]

        total_batches += 1

    # ------------------------------------------------
    # Epoch Summary
    # ------------------------------------------------
    return {
        "loss": total_loss / total_batches,
        "main_loss": total_main_loss / total_batches
    }
