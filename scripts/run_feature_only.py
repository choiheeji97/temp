"""
5-fold cross-validation for the feature-only (quantitative measurements) MLP.

Only the tabular quantitative features (no image data) are used.  The MLP
is trained directly on the measurement columns listed in ``QUANTI_COLUMNS``.

Usage (from project root):
    python scripts/run_feature_only.py

Each fold's outputs are saved under <OUTPUT_BASE>/fold<k>/:
    best_model_feature_only.pt
    best_model_metric_feature_only.json
    class_weights.json
    results_train.csv              – train-split predictions at the best-F1 epoch
    results_val.csv                – val-split predictions at the best-F1 epoch
    test_inference.csv             – test-split predictions from final inference
    metrics_test.json              – full test metrics (AUC, PR-AUC, Acc, Sen, Spe, Pre, F1)
    history_feature_only.csv       – per-epoch metric history
    config.json                    – hyperparameters used for this fold
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import torch

from configs.config import seed_everything, device
from src.model import create_fc_model
from src.train_feature_only import train_feature_only
from src.test import inference_feature_only

# ── paths (edit before running) ──────────────────────────────────────────────
DATASET_CSV = 'path/to/final_dataset.csv'   # must have columns: filename, img_dir, label, fold1-fold5
OUTPUT_BASE = 'outputs/feature_only'
# ─────────────────────────────────────────────────────────────────────────────

QUANTI_COLUMNS = [
    'QMD_sup2', 'QMD_sup1', 'QMD_ref',
    'QMD_inf1', 'QMD_inf2', 'QMD_inf3',
    'QMD_inf4', 'QMD_inf5', 'QMD_inf6', 'QMD_inf7',
]

CFG = {
    'EPOCHS':           50,
    'LEARNING_RATE':    1e-3,
    'BATCH_SIZE':       8,
    'WEIGHT_DECAY':     1e-4,
    'LABEL_SMOOTHING':  0.0,
    'USE_CLASS_WEIGHT': True,
    'SEED':             42,
    'N_LAYERS':         2,
    'FIRST_HIDDEN':     16,
    'HIDDEN_SIZE':      None,   # None → no intermediate hidden layer
    'USE_DROPOUT':      True,
    'DROPOUT_RATE':     0.2,
}


def run_fold(fold):
    """Train and evaluate the feature-only MLP for a single fold."""
    model_ckpt = os.path.join(OUTPUT_BASE, f'fold{fold}')
    os.makedirs(model_ckpt, exist_ok=True)

    seed_everything(CFG['SEED'])

    filelist = pd.read_csv(DATASET_CSV).sort_values('filename').reset_index(drop=True)
    train_df = filelist[filelist[f'fold{fold}'] == 'train'].reset_index(drop=True)
    val_df   = filelist[filelist[f'fold{fold}'] == 'val'].reset_index(drop=True)
    test_df  = filelist[filelist[f'fold{fold}'] == 'test'].reset_index(drop=True)

    # Clamp intermediate hidden size to at most half of first_hidden.
    hidden_sizes = ([] if CFG['HIDDEN_SIZE'] is None
                    else [min(CFG['HIDDEN_SIZE'], CFG['FIRST_HIDDEN'] // 2)])

    fc_model = create_fc_model(
        input_size=len(QUANTI_COLUMNS),
        n_layers=CFG['N_LAYERS'],
        first_hidden=CFG['FIRST_HIDDEN'],
        hidden_sizes=hidden_sizes,
        use_dropout=CFG['USE_DROPOUT'],
        dropout_rate=CFG['DROPOUT_RATE'],
    ).to(device)

    fold_cfg = {**CFG, 'MODEL_CKPT': model_ckpt, 'QUANTI_COLUMNS': QUANTI_COLUMNS}
    with open(os.path.join(model_ckpt, 'config.json'), 'w') as f:
        json.dump(fold_cfg, f, indent=4)

    best_model, best_val_f1 = train_feature_only(
        fc_model=fc_model,
        quanti_columns=QUANTI_COLUMNS,
        train_df=train_df,
        val_df=val_df,
        num_epochs=CFG['EPOCHS'],
        batch_size=CFG['BATCH_SIZE'],
        lr=CFG['LEARNING_RATE'],
        wd=CFG['WEIGHT_DECAY'],
        label_smoothing=CFG['LABEL_SMOOTHING'],
        use_class_weight=CFG['USE_CLASS_WEIGHT'],
        model_ckpt=model_ckpt,
        device=device,
    )

    X_test    = test_df[QUANTI_COLUMNS].values.astype(np.float32)
    y_test    = test_df['label'].values
    path_test = test_df['img_dir'].values

    inference_feature_only(
        fc_model=best_model,
        X_test=X_test,
        y_test=y_test,
        path_test=path_test,
        device=device,
        model_ckpt=model_ckpt,
        batch_size=CFG['BATCH_SIZE'],
        mode='test',
    )


if __name__ == '__main__':
    for fold in range(1, 6):
        print(f'\n{"="*40}\nFold {fold}/5\n{"="*40}')
        run_fold(fold)
