# Multimodal Fundus Image Classification

Code for the paper: binary classification of fundus images using image-only, feature-only, and multimodal fusion approaches, evaluated under 5-fold cross-validation.

## Models

We compare three variants:

- **Image-only** — ResNet18, fine-tuned end-to-end on fundus photographs.
- **Feature-only** — A small MLP that takes only the quantitative measurement (QMD) columns as input.
- **Multimodal** — Frozen ResNet18 features concatenated with QMD columns, followed by a trainable MLP head. This requires a pretrained image-only checkpoint.

Best-epoch selection is based on validation F1 in all cases.

## Repository layout

```
configs/
  config.py                 # seed utilities, device selection

scripts/
  run_image_only.py         # 5-fold CV for the image-only model
  run_feature_only.py       # 5-fold CV for the feature-only model
  run_multimodal.py         # 5-fold CV for the multimodal model
  cv_summary.py             # regenerate cv_summary.csv (all models)
  cv_ensemble.py            # regenerate ensemble outputs (all models)

src/
  dataset.py                # dataset class with per-channel normalisation
  model.py                  # backbone factory + classifier builder
  train_image_only.py       # training loop (image-only)
  train_feature_only.py     # training loop (feature-only)
  train_multimodal.py       # feature extraction + training loop (multimodal)
  test.py                   # shared inference functions
  cv_summary.py             # CV summary logic (imported by run scripts)
  cv_ensemble.py            # ensemble logic (imported by run scripts)
  utils.py                  # class-weight computation
```

## Data format

Prepare a single CSV with the following columns:

- `filename` — unique image identifier (used for sorting)
- `img_dir` — absolute path to each image
- `label` — 0 or 1
- `fold1` through `fold5` — split assignment per fold (`train`, `val`, or `test`)
- `QMD_sup2`, `QMD_sup1`, … — quantitative measurement columns (needed for the feature-only and multimodal pipelines)

## How to run

Set `DATASET_CSV` and `OUTPUT_BASE` at the top of each script, then:

```bash
# 1. Image-only
python scripts/run_image_only.py

# 2. Feature-only
python scripts/run_feature_only.py

# 3. Multimodal (train image-only first — its checkpoints serve as the feature extractor)
#    Point IO_CHECKPOINT_BASE to the image-only output directory.
python scripts/run_multimodal.py
```

After all five folds finish, each script automatically writes `cv_summary.csv` and the ensemble outputs. If you want to regenerate those separately (e.g. after changing a threshold), run:

```bash
python scripts/cv_summary.py
python scripts/cv_ensemble.py
```

## What gets saved

Each fold produces its own directory (`<OUTPUT_BASE>/fold<k>/`) containing the best checkpoint, per-sample predictions for train/val/test splits, test metrics, a training history log, and the hyperparameters and class weights used.

At the model level, three aggregate files are written to `<OUTPUT_BASE>/`:

- `cv_summary.csv` — per-fold and summary (mean, SD) rows for validation and test
- `test_inference_ensemble.csv` — per-sample ensemble predictions across all folds
- `metrics_test_ensemble.json` — ensemble test metrics (AUC, PR-AUC, accuracy, sensitivity, specificity, precision, F1)

## Reproducibility

All experiments fix the random seed (default 42) across Python, NumPy, and PyTorch (CPU and CUDA). DataLoader workers are seeded via `seed_worker` and per-worker `torch.Generator` instances.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision, scikit-learn, pandas, numpy, opencv-python, tqdm
