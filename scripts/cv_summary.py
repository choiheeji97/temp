"""
Cross-validation summary for all three model pipelines.

For each model (image_only, feature_only, multimodal), this script loads
per-fold results_val.csv and test_inference.csv, computes per-fold and
aggregate (mean ± SD) metrics, and saves a cv_summary.csv inside each
model's output directory.

Usage (from project root):
    python scripts/cv_summary.py

Expected directory layout (produced by the run_*.py scripts):
    outputs/<model>/fold1/results_val.csv
    outputs/<model>/fold1/test_inference.csv
    ...
    outputs/<model>/fold5/results_val.csv
    outputs/<model>/fold5/test_inference.csv

Outputs:
    outputs/<model>/cv_summary.csv  – per-fold rows + mean / SD / mean±SD rows
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
)

# ── output directories for each pipeline ─────────────────────────────────────
MODELS = {
    'image_only':   'outputs/image_only',
    'feature_only': 'outputs/feature_only',
    'multimodal':   'outputs/multimodal',
}
N_FOLDS = 5
METRICS = ['acc', 'auc', 'prauc', 'sen', 'spe', 'ppv', 'f1']
# ─────────────────────────────────────────────────────────────────────────────


def calc_metrics(df, model_name, fold, split):
    """Compute binary classification metrics for one fold/split DataFrame."""
    return {
        'model':     model_name,
        'datasplit': split,
        'fold':      fold,
        'acc':       accuracy_score(df['label'], df['pred']),
        'auc':       roc_auc_score(df['label'], df['prob']),
        'prauc':     average_precision_score(df['label'], df['prob']),
        'sen':       recall_score(df['label'], df['pred'], pos_label=1, zero_division=0),
        'spe':       recall_score(df['label'], df['pred'], pos_label=0, zero_division=0),
        'ppv':       precision_score(df['label'], df['pred'], zero_division=0),
        'f1':        f1_score(df['label'], df['pred'], zero_division=0),
    }


def get_metrics(model_name, output_base, fold):
    """Load val and test CSVs for one fold and return a two-row metrics DataFrame."""
    fold_dir = os.path.join(output_base, f'fold{fold}')
    val  = pd.read_csv(os.path.join(fold_dir, 'results_val.csv'))
    test = pd.read_csv(os.path.join(fold_dir, 'test_inference.csv'))
    return pd.DataFrame([
        calc_metrics(val,  model_name, fold, 'val'),
        calc_metrics(test, model_name, fold, 'test'),
    ])


def cv_results(model_name, output_base, n_folds=N_FOLDS):
    """Build a summary DataFrame of n-fold CV results for one model.

    Returned DataFrame rows:
        fold 1…n  – per-fold val and test metrics (numeric)
        mean      – column-wise mean across folds, per split
        sd        – column-wise standard deviation, per split
        mean±sd   – formatted string "X.XX±X.XX" for each metric, per split
    """
    # Collect per-fold metrics
    res_all = pd.concat(
        [get_metrics(model_name, output_base, fold) for fold in range(1, n_folds + 1)],
        axis=0,
    ).reset_index(drop=True)

    # Aggregate mean and SD grouped by datasplit (val / test)
    summary = res_all.groupby('datasplit')[METRICS].agg(['mean', 'std'])

    # Build mean±sd formatted string row
    mean_sd_df = pd.DataFrame()
    for m in METRICS:
        mean_sd_df[m] = (
            summary[(m, 'mean')].map(lambda x: f"{x:.2f}")
            + '±'
            + summary[(m, 'std')].map(lambda x: f"{x:.2f}")
        )
    mean_sd_df = mean_sd_df.reset_index()
    mean_sd_df.insert(0, 'fold', 'mean±sd')
    mean_sd_df.insert(0, 'model', model_name)

    # Build numeric mean and SD rows
    mean_df = summary.xs('mean', axis=1, level=1).reset_index()
    mean_df.insert(0, 'fold', 'mean')
    mean_df.insert(0, 'model', model_name)

    sd_df = summary.xs('std', axis=1, level=1).reset_index()
    sd_df.insert(0, 'fold', 'sd')
    sd_df.insert(0, 'model', model_name)

    # Concatenate all rows; sort so 'val' appears before 'test'
    final_df = pd.concat(
        [res_all, mean_df, sd_df, mean_sd_df],
        ignore_index=True,
    ).sort_values('datasplit', ascending=False).reset_index(drop=True)

    return final_df


if __name__ == '__main__':
    for model_name, output_base in MODELS.items():
        if not os.path.isdir(output_base):
            print(f'[{model_name}] Directory not found: {output_base} — skipping.')
            continue

        print(f'[{model_name}] Building CV summary ...')
        df = cv_results(model_name, output_base)

        save_path = os.path.join(output_base, 'cv_summary.csv')
        df.to_csv(save_path, index=False)
        print(f'[{model_name}] Saved → {save_path}')
        print(df.to_string(index=False))
        print()
