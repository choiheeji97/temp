# Multimodal Fundus Image Classification

This repository contains the training and evaluation code for binary fundus image classification using three model variants: image-only CNN, feature-only MLP (quantitative measurements), and a multimodal fusion model.

## Overview

| Model | Description |
|-------|-------------|
| **Image-only** | ResNet18 fine-tuned end-to-end on fundus images |
| **Feature-only** | Lightweight MLP trained on quantitative measurements (QMD columns) |
| **Multimodal** | Concatenates frozen ResNet18 image features with quantitative measurements, then trains an MLP head |

All three models are evaluated under 5-fold cross-validation.  Best-epoch selection is based on validation F1 score.

---

## Directory Structure

```
.
├── configs/
│   └── config.py              # Seed utilities and device selection
├── scripts/
│   ├── run_image_only.py      # Train & evaluate image-only model (5-fold CV)
│   ├── run_feature_only.py    # Train & evaluate feature-only model (5-fold CV)
│   ├── run_multimodal.py      # Train & evaluate multimodal model (5-fold CV)
│   ├── cv_summary.py          # Standalone: regenerate cv_summary.csv for all models
│   └── cv_ensemble.py         # Standalone: regenerate ensemble outputs for one model
└── src/
    ├── dataset.py              # CustomDataset with per-channel normalisation
    ├── model.py                # Backbone factory + FC model builder
    ├── train_image_only.py     # Training loop for the image-only CNN
    ├── train_feature_only.py   # Training loop for the feature-only MLP
    ├── train_multimodal.py     # Feature extraction + training loop for multimodal MLP
    ├── test.py                 # Inference functions shared by all three pipelines
    ├── cv_summary.py           # CV summary logic imported by run_*.py
    ├── cv_ensemble.py          # Ensemble logic imported by run_feature_only.py
    └── utils.py                # Class-weight computation helpers
```

---

## Data Format

The dataset CSV must contain the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Unique image identifier (used for sorting) |
| `img_dir` | Absolute path to the image file |
| `label` | Binary label (0 or 1) |
| `fold1` … `fold5` | Split assignment per fold: `'train'`, `'val'`, or `'test'` |
| `QMD_sup2`, `QMD_sup1`, … | Quantitative measurement columns (required for feature-only and multimodal pipelines) |

---

## Usage

Edit the `DATASET_CSV` and `OUTPUT_BASE` paths at the top of each script, then run from the project root.

### 1. Image-only model

```bash
python scripts/run_image_only.py
```

### 2. Feature-only model

```bash
python scripts/run_feature_only.py
```

### 3. Multimodal model

The image-only model **must be trained first** because its checkpoints are used as the feature extractor.  Update `IO_CHECKPOINT_BASE` to point to the image-only output directory.

```bash
python scripts/run_multimodal.py
```

### Post-training outputs (automatic)

`cv_summary.csv` and ensemble files are generated **automatically** at the end of each run script — no extra step needed.

| Script | Auto-generates |
|--------|---------------|
| `run_image_only.py` | `outputs/image_only/cv_summary.csv` |
| `run_feature_only.py` | `outputs/feature_only/cv_summary.csv` · `test_inference_ensemble.csv` · `metrics_test_ensemble.json` |
| `run_multimodal.py` | `outputs/multimodal/cv_summary.csv` |

To regenerate these files independently (e.g. after adding a fold or changing thresholds):

```bash
python scripts/cv_summary.py   # cv_summary.csv for all three models
python scripts/cv_ensemble.py  # ensemble outputs (feature_only by default; edit OUTPUT_BASE for others)
```

---

## Output Files

### Per-fold outputs (`<OUTPUT_BASE>/fold<k>/`)


| File | Description |
|------|-------------|
| `best_model*.pt` | Best model state dict |
| `best_model_metric*.json` | Validation metrics at the best-F1 epoch (includes `train_prauc` and `val_prauc`) |
| `config.json` | Hyperparameters used for this fold |
| `class_weights.json` | Complement-frequency class weights used in training |
| `dataset_statistics.json` | Training-set per-channel mean and std (image pipelines only) |
| `results_train.csv` | Per-sample predictions for the **training** split at the best-F1 epoch |
| `results_val.csv` | Per-sample predictions for the **validation** split at the best-F1 epoch |
| `test_inference.csv` | Per-sample predictions for the **test** split |
| `metrics_test.json` | Full test-set metrics (see below) |
| `history*.csv` | Per-epoch training and validation metric history |

### Per-model aggregate outputs (`<OUTPUT_BASE>/`)

All three pipelines produce:

| File | Description |
|------|-------------|
| `cv_summary.csv` | Per-fold rows + mean / SD / mean±SD rows for val and test splits |

Feature-only pipeline additionally produces:

| File | Description |
|------|-------------|
| `test_inference_ensemble.csv` | Per-sample ensemble predictions merged from all 5 folds |
| `metrics_test_ensemble.json` | Ensemble test-set metrics (AUC, PR-AUC, Acc, Sen, Spe, Pre, F1) |

### Metrics saved in `metrics_test.json`

| Key | Definition |
|-----|-----------|
| `auc` | Area under the ROC curve (AUROC) |
| `prauc` | Area under the Precision–Recall curve (average precision) |
| `acc` | Overall accuracy |
| `sensitivity` | Recall for the positive class (true positive rate) |
| `specificity` | Recall for the negative class (true negative rate) |
| `precision` | Positive predictive value |
| `f1` | Harmonic mean of precision and sensitivity |

---

## Reproducibility

All experiments use a fixed random seed (`SEED = 42` by default) applied to Python `random`, NumPy, PyTorch CPU/CUDA, and DataLoader workers via `seed_everything` and `seed_worker` in `configs/config.py`.  DataLoader generators are initialised inline with `torch.Generator().manual_seed(seed)`.

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- scikit-learn
- pandas, numpy, opencv-python, tqdm
