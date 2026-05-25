import torch
from utils.eval import evaluate_one_epoch
from torch.utils.data import DataLoader
from utils.config import parse_args, load_config
from model.model import MoEMultimodalEmotionModel
from utils.logger import setup_logger
from datasets.MosiDataset import MOSIDataset
from losses.regression_loss import RegressionMoELoss
from losses.classification_loss import ClassificationMoELoss
from losses.contrastive_loss import EmotionContrastiveLoss

def inference(model, dataloader, device):
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())

    return torch.cat(all_outputs, dim=0)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logger(config.log_path)
    # ==================== Load Best Model & Test ====================
    logger.info(f"\nLoading best checkpoint from {config.ckpt_path} ...")
        # ==================== Model ====================
    model = MoEMultimodalEmotionModel(config).to(config.device)


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

    ckpt = torch.load(config.ckpt_path, map_location=config.device)

    # run_stress_test.py

    # 1. 这里的 drop_rate 是你训练好的模型所对应的参数
    # 比如你现在的模型是用 drop_rate=0.3 训练出来的
    train_drop_rate = 0.3 

    # 2. 我们要扫描的噪声等级
    test_noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = {}

    for nl in test_noise_levels:
        print(f"🚀 Running Stress Test: Noise Level = {nl}")
        
        # 加载测试集 (注意：测试集的 drop_rate 此时可以设为 0，因为我们只测噪声鲁棒性)
        # 或者如果你想测“既有缺失又有噪声”，也可以设为 0.1
        test_dataset = MOSIDataset(config.data_path, split='test', drop_rate=0.0, noise_level=nl)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        test_metrics = evaluate_one_epoch(
            config, model, test_loader,
            loss_fn=loss_fn,
            contrastive_loss_fn=contrastive_fn,
            use_contrastive=config.use_contrastive,
            device=config.device
        )

        print("\n===== [Test Result] =====")
        print(f"Noise Level: {nl}")
        print(test_metrics)