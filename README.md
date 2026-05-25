# QA-MoE: Mixture-of-Experts Multimodal Emotion Analysis

A multimodal emotion analysis framework based on Mixture-of-Experts (MoE), supporting sentiment regression and classification on CMU-MOSI / CMU-MOSEI and MIntRec datasets. The model fuses text, audio, and visual modalities with modality quality gating and contrastive learning.

---

## Project Structure

```
QA-MoE/
├── train.py                  # Training entry point
├── inference.py              # Inference entry point
├── configs/
│   └── mosi.yaml             # Example config for MOSI
├── model/
│   └── model.py              # MoEMultimodalEmotionModel
├── datasets/
│   ├── MosiDataset.py        # MOSI / MOSEI dataset loader
│   └── base.py               # MIntRec DataManager
├── losses/
│   ├── regression_loss.py    # RegressionMoELoss
│   ├── classification_loss.py# ClassificationMoELoss
│   └── contrastive_loss.py   # EmotionContrastiveLoss
├── utils/
│   ├── trainer.py
│   ├── eval.py
│   ├── checkpoint.py
│   ├── early_stopping.py
│   ├── logger.py
│   └── config.py
└── checkpoints/              # Saved model checkpoints
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- transformers
- numpy
- scikit-learn

Install dependencies:

```bash
pip install torch transformers numpy scikit-learn
```

---

## Data Preparation

This project uses the **CMU-MOSI** / **CMU-MOSEI** aligned features (`.pkl` format) and **MIntRec**.

Set the correct path in your config file:

```yaml
data_path: "path/to/CMU-MOSI/Processed/aligned_50.pkl"
```

---

## Training

```bash
python train.py --config configs/mosi.yaml
```

You can switch datasets by changing the config file. A MIntRec example:

```bash
python train.py --config configs/mintrec.yaml
```

---

## Inference

```bash
python inference.py --config configs/mosi.yaml
```

---

## Configuration

All hyperparameters are controlled via a YAML config file. Key options:

| Parameter | Description | Default |
|---|---|---|
| `dataset` | Dataset name: `MOSI`, `MOSEI`, `MIntRec` | `MOSI` |
| `data_path` | Path to processed dataset file | — |
| `epochs` | Number of training epochs | `30` |
| `batch_size` | Batch size | `16` |
| `lr` | Learning rate | `1e-4` |
| `hidden_dim` | Encoder output dimension | `128` |
| `num_experts` | Number of MoE experts | `8` |
| `top_k` | Number of experts selected per sample | `3` |
| `use_quality` | Enable modality quality gating | `true` |
| `use_contrastive` | Enable contrastive loss | `true` |
| `drop_rates` | Modality drop rates for robustness testing | `[0.0 ~ 1.0]` |
| `noise_levels` | Gaussian noise levels for robustness testing | `[0.0 ~ 0.7]` |
| `ckpt_path` | Path to save / load checkpoint | — |
| `log_path` | Path to training log | `./logs/train.log` |

Full example config:

```yaml
dataset: "MOSI"
data_path: "../../data/CMU-MOSI/Processed/aligned_50.pkl"

seed: 42
device: "cuda"

batch_size: 16
epochs: 30
lr: 0.0001

hidden_dim: 128
num_experts: 8
top_k: 3
use_quality: true

use_contrastive: true
contrastive_freeze_epoch: 5

text_input_dim: 768
audio_input_dim: 5
vision_input_dim: 20

reg_weight: 1.0
quality_weight: 0.5
contrastive_weight: 0.05
entropy_weight: 0.0
load_balance_weight: 0.01

drop_rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_levels: [0.0, 0.1, 0.2, 0.3]

log_path: "./logs/train.log"
ckpt_path: "./checkpoints/best_model.pth"
```

---

## Input Features

| Modality | Feature | Dimension |
|---|---|---|
| Text | BERT | 768 |
| Audio | COVAREP | 5 |
| Vision | FACET | 20 |

---

## Model Overview

- **Text Encoder**: BERT (fine-tunable, frozen for the first 5 epochs)
- **Audio / Vision Encoder**: Projection layers
- **Fusion**: Mixture-of-Experts with top-k routing and modality quality gating
- **Expert**: Transformer-based (hidden dim 256, 4 heads, 2 layers)
- **Loss**: Regression (MAE) + quality gating loss + contrastive loss + MoE load balance regularization
- **Training**: AdamW optimizer, ReduceLROnPlateau scheduler, EarlyStopping (patience=7)

---

## Robustness Evaluation

The test phase automatically evaluates across all combinations of `drop_rates` and `noise_levels` defined in the config, allowing systematic robustness analysis under missing modality and noisy input conditions.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{zhu2026qamoecontinuousreliabilityspectrum,
      title={QA-MoE: Towards a Continuous Reliability Spectrum with Quality-Aware Mixture of Experts for Robust Multimodal Sentiment Analysis}, 
      author={Yitong Zhu and Yuxuan Jiang and Guanxuan Jiang and Bojing Hou and Peng Yuan Zhou and Ge Lin Kan and Yuyang Wang},
      year={2026},
      eprint={2604.05704},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.05704}, 
}
```

---

## License

MIT License
