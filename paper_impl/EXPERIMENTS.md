# QA-MoE calibration ablations (experimental research branch)

This branch exp/calibration-ablation-20261014 is separate from the prior paper-faithful-qa-moe-20261014 baseline and master. The previous checkpoint is untouched.
The default model remains unmodified without optional flags.
Adding variance floor, auxiliary Huber, or router balance changes the paper loss and must be reported as an experimental ablation.

## Diagnose existing best checkpoint

Run from repository root:

    python -m paper_impl.diagnose --checkpoint runs/mosi_clean/best.pt --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --split valid

    python -m paper_impl.reliability_probe --checkpoint runs/mosi_clean/best.pt --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --batch-size 16

This probe applies isolated text token masks, audio feature Gaussian noise, and vision feature Gaussian noise at strengths 0, 0.3, 0.7 and 1.0. It logs modality reliability changes and validation MAE. This is not a ground-truth quality calibration test.

## Validation-only controlled training experiments

B. NLL with smooth log-variance floor -1.0:

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --protocol clean --batch-size 8 --epochs 30 --seed 1111 --amp --logvar-floor -1.0 --skip-test-after-training --output runs/mosi_floor_m1

C. Same floor, auxiliary Huber loss weight 0.3:

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --protocol clean --batch-size 8 --epochs 30 --seed 1111 --amp --logvar-floor -1.0 --aux-huber-weight 0.3 --skip-test-after-training --output runs/mosi_floor_huber03

D. Optional router balance penalty after analyzing first two experiments:

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --protocol clean --batch-size 8 --epochs 30 --seed 1111 --amp --logvar-floor -1.0 --aux-huber-weight 0.3 --router-balance-weight 0.01 --skip-test-after-training --output runs/mosi_floor_huber_balance

Best checkpoints are selected by validation MAE. Training logs distinguish raw NLL and total objective. Diagnose each best checkpoint on validation before selecting a configuration. The original test score must NOT be used to choose hyperparameters.

Smooth floor: log_var = floor + softplus(raw_log_var - floor). At floor=-1, sigma >= exp(-0.5) ~= 0.607. This can help reduce extreme overconfidence but cannot guarantee a lower MAE.

## After validating and choosing a configuration

    python -m paper_impl.train --dataset mosi --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --mode eval --checkpoint runs/mosi_floor_huber03/best.pt --batch-size 8 --amp --output runs/mosi_floor_huber03_test

The evaluation loader gets logvar_floor from checkpoint arguments.

IMPORTANT: GitHub code was pushed and reviewed statically; no server/GPU execution of this experimental branch has been performed. Compare results over multiple random seeds and keep reported paper, reconstructed baseline and experimental ablation metrics distinct.
