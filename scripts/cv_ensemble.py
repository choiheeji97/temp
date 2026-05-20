"""
Ensemble test-set predictions across all 5 folds for a given model pipeline.

For each fold, loads test_inference.csv from the fold's output directory,
renames the prob and pred columns to include the fold index (prob_fold1,
pred_fold1, …), then merges all folds on image_path + label (inner join,
keeping only samples present in every fold's test set).

The ensemble probability is the row-wise mean of per-fold probabilities.
The ensemble hard prediction uses a 0.5 threshold on that mean probability.

Outputs written to OUTPUT_BASE:
    test_inference_ensemble.csv  – per-sample ensemble predictions with
                                   individual fold columns for traceability
    metrics_test_ensemble.json   – ensemble performance metrics

Usage (from project root):
    python scripts/cv_ensemble.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score,
)

# ── paths (edit before running) ──────────────────────────────────────────────
OUTPUT_BASE = 'outputs/feature_only'
N_FOLDS     = 5
THRESHOLD   = 0.5   # decision boundary for hard ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────


def load_fold_inference(output_base, fold):
    """Load test_inference.csv for one fold; rename prob/pred with fold suffix."""
    path = os.path.join(output_base, f'fold{fold}', 'test_inference.csv')
    df = pd.read_csv(path)
    df = df.rename(columns={
        'prob': f'prob_fold{fold}',
        'pred': f'pred_fold{fold}',
    })
    return df


def build_ensemble(output_base, n_folds, threshold=THRESHOLD):
    """Merge per-fold inferences and compute ensemble probability and prediction.

    Parameters
    ----------
    output_base : str
        Root output directory (e.g. 'outputs/feature_only').
    n_folds : int
        Number of folds to aggregate.
    threshold : float
        Decision threshold applied to the mean probability.

    Returns
    -------
    pd.DataFrame with columns:
        image_path, label,
        prob_fold1 … prob_foldN, pred_fold1 … pred_foldN,
        prob_ensemble, pred_ensemble
    """
    # Load fold 1 as the base, then iteratively merge remaining folds
    merged = load_fold_inference(output_base, 1)

    for fold in range(2, n_folds + 1):
        df_fold = load_fold_inference(output_base, fold)
        merged = merged.merge(
            df_fold,
            on=['image_path', 'label'],
            how='inner',   # keep only samples present in all folds
        )

    # Compute ensemble probability as the mean across folds
    prob_cols = [f'prob_fold{k}' for k in range(1, n_folds + 1)]
    pred_cols = [f'pred_fold{k}' for k in range(1, n_folds + 1)]

    merged['prob_ensemble'] = merged[prob_cols].mean(axis=1)
    merged['pred_ensemble'] = (merged['prob_ensemble'] >= threshold).astype(int)

    # Reorder columns: identity → per-fold probs → per-fold preds → ensemble
    col_order = (
        ['image_path', 'label']
        + prob_cols
        + pred_cols
        + ['prob_ensemble', 'pred_ensemble']
    )
    return merged[col_order]


def compute_ensemble_metrics(df):
    """Compute binary classification metrics for the ensemble predictions."""
    trues = df['label']
    probs = df['prob_ensemble']
    preds = df['pred_ensemble']
    return {
        'auc':         roc_auc_score(trues, probs),
        'prauc':       average_precision_score(trues, probs),
        'acc':         accuracy_score(trues, preds),
        'sensitivity': recall_score(trues, preds, pos_label=1, zero_division=0),
        'specificity': recall_score(trues, preds, pos_label=0, zero_division=0),
        'precision':   precision_score(trues, preds, zero_division=0),
        'f1':          f1_score(trues, preds, zero_division=0),
    }


if __name__ == '__main__':
    print(f'Building {N_FOLDS}-fold ensemble from: {OUTPUT_BASE}')

    ensemble_df = build_ensemble(OUTPUT_BASE, N_FOLDS, THRESHOLD)
    print(f'Ensemble test set size: {len(ensemble_df)} samples')

    # Save per-sample ensemble predictions
    csv_path = os.path.join(OUTPUT_BASE, 'test_inference_ensemble.csv')
    ensemble_df.to_csv(csv_path, index=False)
    print(f'Saved → {csv_path}')

    # Compute and save ensemble metrics
    metrics = compute_ensemble_metrics(ensemble_df)
    print('Ensemble metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

    json_path = os.path.join(OUTPUT_BASE, 'metrics_test_ensemble.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Saved → {json_path}')
