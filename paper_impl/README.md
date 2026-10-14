# QA-MoE paper-oriented reconstruction

Status: independent paper-based implementation, NOT verified official source code and NOT a guarantee of the ACL 2026 paper's reported scores. The preexisting code is untouched. Only paper_impl/ is new.

## Implementation coverage

- Eq. (3-5): separate learned Gaussian means and softplus variances for text/audio/vision.
- Eq. (6): per-modality continuous reliability from predicted variance, forced to zero for missing modalities.
- Eq. (7): semantic GLU expert routing for EACH modality, eight experts and actual sparse top-3 compute; learned global prior for low-reliability fallback.
- Eqs. (8-10): regression value and log-variance heads with heteroscedastic Gaussian NLL.
- Clean aligned train for paper Table 1; spectrum-aware train with per-batch Uniform(0,1) lambda and eta as in Appendix A.1.
- No sentiment-label normalization: labels stay on original [-3,3] scale.
- Deterministic fixed-condition test grid; checkpoint loading; additional CPU self-check.

IMPORTANT: The paper omits some architecture details. Our audio/vision encoder is a BiGRU, text uses BERT masked mean, and fusion is a small MLP. Masked text corruption uses BERT [MASK] token. These are reconstruction decisions, not demonstrated matches with the authors' hidden implementation. No original author weights are bundled. Supports aligned MOSI/MOSEI only, not unaligned_50 or MIntRec.

## Environment: Linux with RTX 2080 Ti 22GB

Create a fresh environment in your cloned repository:

    conda create -n qa-moe-paper python=3.10 -y
    conda activate qa-moe-paper
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
    pip install 'transformers>=4.36,<4.45' 'numpy<2' scikit-learn tqdm

The initial BERT-base-uncased download needs access to Hugging Face. If the server is offline, pass --bert-path /local/bert-base-uncased. The installed NVIDIA driver must support the chosen CUDA runtime. On PyTorch 2.3+, the FP16 --amp flag is optional.

## Check paths and self-check

Run from repository root:

    nvidia-smi
    ls -lh /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl
    ls -lh /data4t/lyx/datasets/MOSEI/Processed/aligned_50*.pkl
    python -m paper_impl.selfcheck

First smoke test (two optimizer steps; NOT a paper-comparable run):

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --epochs 1 --max-steps 2 --batch-size 2 --output runs/mosi_smoke

Peak allocated GPU memory prints after each epoch. If out of memory, start with batch-size 2 or --amp. --freeze-bert can reduce VRAM but is NOT a strict paper configuration.

## Reproduce paper Table 1 (clean/aligned datasets)

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --protocol clean --batch-size 8 --epochs 30 --seed 1111 --amp --output runs/mosi_clean

    python -m paper_impl.train --dataset mosei --data /data4t/lyx/datasets/MOSEI/Processed --protocol clean --batch-size 8 --epochs 30 --seed 1111 --amp --output runs/mosei_clean

MOSEI directory resolves aligned_50.pkl and then aligned_50-001.pkl. If neither exists or multiple other aligned_50*.pkl exist, pass exact file path.

Outputs: best.pt, history.jsonl, test_clean.json. Results use Acc_7, Acc_2_nonzero, F1_nonzero, Acc_2_including_zero, MAE, Corr. First compare predictions and references on the SAME metrics convention and source features. Paper aligned benchmark target: MOSI Acc-7 53.6, MAE 0.579; MOSEI Acc-7 58.4, MAE 0.477. These are not guaranteed.

## Spectrum-aware experiment

Appendix A.1: each training batch samples independently lambda and eta from Uniform(0,1). Audio/visual noise uses training-set reference std; text word-token masking uses BERT MASK. Missingness is sampled per example/modality at batch-level eta.

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --protocol spectrum --batch-size 8 --epochs 30 --seed 1111 --amp --output runs/mosi_spectrum

To evaluate an existing checkpoint at fixed missing rates and noise levels from 0 through 0.7:

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --mode eval --checkpoint runs/mosi_spectrum/best.pt --eval-grid --batch-size 8 --amp --output runs/mosi_grid

The 64-condition grid takes a while: 8 x 8 full test evaluations. Set --grid-step 0.2 for a smaller development grid. The grid is saved as spectrum_grid.csv. Use --mode eval without --eval-grid for clean test only.

## Caveats

1. This independent reconstruction is preliminary until run with your actual dataset/GPU and compared to ablations in the ACL paper.
2. Frozen BERT, smaller batch, changed preprocessing or feature variants are not numerically identical to the paper.
3. Main Table 1 clean-training and Appendix Spectrum-training use DIFFERENT checkpoints.
4. Keep raw sentiment labels unchanged. The original upstream code normalizes labels and fails to undo scaling during evaluation.
5. Negative/positive binary metrics can include/exclude zero: this implementation logs both, with F1 excluding zero.
6. Dataset file checksums and shapes should be recorded to make results reproducible.
