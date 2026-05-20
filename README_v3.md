# Multimodal Fundus Image Classification

Code for binary classification of fundus images using image-only, feature-only, and multimodal fusion approaches, evaluated under 5-fold cross-validation.

## Models

- **Image-only**: ResNet18 fine-tuned end-to-end on fundus photographs.
- **Feature-only**: A small MLP trained on quantitative measurement (QMD) columns only.
- **Multimodal**: Frozen ResNet18 features concatenated with QMD columns, followed by a trainable MLP head. Requires a pretrained image-only checkpoint.

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

- `filename`: unique image identifier
- `img_dir`: absolute path to each image
- `label`: 0 or 1
- `fold1` through `fold5`: split assignment per fold (`train`, `val`, or `test`)
- `QMD_*` columns: quantitative measurements (needed for feature-only and multimodal)

## How to run

Set `DATASET_CSV` and `OUTPUT_BASE` at the top of each script, then:

```bash
python scripts/run_image_only.py
python scripts/run_feature_only.py

# Multimodal requires image-only checkpoints.
# Set IO_CHECKPOINT_BASE to the image-only output directory.
python scripts/run_multimodal.py
```

After all five folds finish, CV summary and ensemble outputs are generated automatically. To regenerate them separately:

```bash
python scripts/cv_summary.py
python scripts/cv_ensemble.py
```

## Reproducibility

All experiments fix the random seed (default 42) across Python, NumPy, and PyTorch (CPU/CUDA).

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision, scikit-learn, pandas, numpy, opencv-python, tqdm
