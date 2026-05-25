import torch
from torch.utils.data import DataLoader
from model.model import MoEMultimodalEmotionModel
from datasets.MosiDataset import MOSIDataset

from losses.regression_loss import RegressionMoELoss
from losses.classification_loss import ClassificationMoELoss
from losses.contrastive_loss import EmotionContrastiveLoss

from utils.trainer import train_one_epoch
from utils.eval import evaluate_one_epoch
from utils.checkpoint import save_checkpoint
from utils.logger import setup_logger
from utils.config import parse_args, load_config

# ========== 新增 EarlyStopping ==========
from utils.early_stopping import EarlyStopping

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(config.log_path)
    logger.info("Loaded config from %s", args.config)

    # ==================== Dataset ====================
    if config.dataset in ["MOSI", "MOSEI"]:
        train_dataset = MOSIDataset(config.data_path, split="train", drop_rate=0.1, noise_level=config.noise_level)
        val_dataset   = MOSIDataset(config.data_path, split="valid", drop_rate=0.1, noise_level=config.noise_level)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    elif config.dataset == "MIntRec":
        from datasets.base import DataManager
        dm = DataManager(config)
        train_loader = dm.mm_dataloader['train']
        val_loader   = dm.mm_dataloader['dev']
        test_loader  = dm.mm_dataloader['test']

    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


    # ==================== Model ====================
    model = MoEMultimodalEmotionModel(config).to(config.device)

    # 在定义优化器时：
    optimizer = torch.optim.AdamW([
        # BERT 学习率极低
        {'params': model.text_encoder.bert.parameters(), 'lr': 1e-5}, 
        # 其他层正常学习率
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': config.lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.5, patience=3,
        # verbose=True
    )

    # ==================== Loss ====================
    if config.dataset in ["MOSI", "MOSEI"]:
        loss_fn = RegressionMoELoss(
            reg_weight=config.reg_weight,
            quality_weight=config.quality_weight,
            entropy_weight=config.entropy_weight
        )
    elif config.dataset == "MIntRec":
        loss_fn = ClassificationMoELoss(
            cls_weight=config.cls_weight
        )
    contrastive_fn = EmotionContrastiveLoss() if config.use_contrastive else None

    # ==================== EarlyStopping ====================
    early_stopper = EarlyStopping(
        patience=7,                # 连续7次验证集指标无提升则早停
        mode="min",                # 'min' 适用于 val_loss 或 MAE
        delta=1e-4,                # 改进阈值
        save_path=config.ckpt_path # 保存最佳模型
    )

    # ==================== Training Loop ====================
    for epoch in range(config.epochs):
        logger.info(f"\n[Epoch {epoch+1}/{config.epochs}]")

        if epoch < 5: 
            for param in model.text_encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.text_encoder.parameters():
                param.requires_grad = True

        # Optional freeze contrastive learning
        if config.use_contrastive and epoch > config.contrastive_freeze_epoch:
            loss_fn.contrastive_weight = 0.1
            logger.info(f"[Epoch {epoch}] Contrastive learning frozen.")

        # ----- Train -----
        train_metrics = train_one_epoch(
            config=config,
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            contrastive_loss_fn=contrastive_fn,
            device=config.device,
            use_contrastive=config.use_contrastive
        )
        logger.info("Train: %s", train_metrics)

        # ----- Validate -----
        val_metrics = evaluate_one_epoch(
            config, model, val_loader,
            loss_fn=loss_fn,
            contrastive_loss_fn=contrastive_fn,
            use_contrastive=config.use_contrastive,
            device=config.device
        )
        logger.info("Eval : %s", val_metrics)

        # Step scheduler based on val loss
        scheduler.step(val_metrics["MAE"])

        # ----- Early Stopping -----
        early_stopper(val_metrics["MAE"], model, optimizer, epoch)

        # If early stopping triggered → break
        if early_stopper.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # ==================== Load Best Model & Test ====================
    logger.info(f"\nLoading best checkpoint from {config.ckpt_path} ...")

    # 加载阶段：
    ckpt = torch.load(config.ckpt_path, map_location=config.device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        model.load_state_dict(ckpt)

    for drop_rate in config.drop_rates:
        # test_dataset  = MOSIDataset(config.data_path, split="test", drop_rate=drop_rate)
        # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        # test_metrics = evaluate_one_epoch(
        #     config, model, test_loader,
        #     loss_fn=loss_fn,
        #     contrastive_loss_fn=contrastive_fn,
        #     use_contrastive=config.use_contrastive,
        #     device=config.device
        # )

        # print("\n===== [Test Result] =====")
        # print(f"Drop Rate: {drop_rate}")
        # print(test_metrics)

        for noise_level in config.noise_levels:
            test_dataset  = MOSIDataset(config.data_path, split="test", drop_rate=drop_rate, noise_level=noise_level)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            test_metrics = evaluate_one_epoch(
                config, model, test_loader,
                loss_fn=loss_fn,
                contrastive_loss_fn=contrastive_fn,
                use_contrastive=config.use_contrastive,
                device=config.device
            )

            print("\n===== [Test Result] =====")
            print(f"Drop Rate: {drop_rate} Noise Level: {noise_level}")
            print(test_metrics)