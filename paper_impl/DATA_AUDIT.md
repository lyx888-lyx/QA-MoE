# Further diagnosis after MOSI reliability and stabilization runs

The MOSI experiment's printed shapes were **Audio=5** and **Vision=20**, while
the QA-MoE ACL 2026 Appendix A.3 claims COVAREP=74 and Facet=35.
This is a potentially critical input-feature mismatch. The correct next step
is a data audit, NOT extensive hyperparameter search.

All commands below are validation-only.

## 1. Inspect every locally available MOSI/MOSEI processed feature version

    python -m paper_impl.audit_data --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl
    python -m paper_impl.audit_data --data /data4t/lyx/datasets/MOSI/Processed/unaligned_50.pkl
    python -m paper_impl.audit_data --data /data4t/lyx/datasets/MOSEI/Processed

This reports each split's feature shapes, labels and text schema.
Beware: pickle is executable serialization; use trusted datasets only.
A matching 74/35 shape still does not prove a matching extraction pipeline.
The training code dynamically uses the dimensions in the file; it does not
convert 5/20 features into 74/35 merely by specifying 74/35 in configuration.
Do not feed an "unaligned" file to the aligned training loader without
implementing and validating that alignment protocol.

## 2. Validate whether the prediction relies on audio/vision at all

    python -m paper_impl.modality_ablation --checkpoint runs/mosi_clean/best.pt --data /data4t/lyx/datasets/MOSI/Processed/aligned_50.pkl --batch-size 16

The script compares clean predictions, a modality permuted between samples,
and a zero+missing modality intervention on validation only. Interpret as
stress/sensitivity tests, NOT ground-truth modality importance.

## 3. Stabilization experiments already performed

A NLL baseline, B log-variance floor=-1, C floor=-1 + Huber=0.3:

- A best validation MAE ~0.6865 / best epoch 16, NLL ~11.12.
- B best validation MAE ~0.6856 / epoch 16, NLL ~0.81.
- C best validation MAE ~0.6829 / epoch 10, NLL ~0.81.

Stabilizing variance does NOT materially improve sentiment MAE in the reported
single-seed results. Do not use test data for hyperparameter selection.
Verify input feature provenance before implementing further loss variants.
