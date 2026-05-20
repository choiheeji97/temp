"""
Inference functions for the three model variants.

Each function loads the best saved checkpoint, runs a forward pass over
the provided data, computes a comprehensive set of binary-classification
metrics, and writes two files to ``model_ckpt``:

* ``test_inference.csv``  – per-sample columns: image_path, prob, pred, label
* ``metrics_test.json``   – AUC, PR-AUC, accuracy, sensitivity, specificity,
                             precision, F1

The ``_compute_metrics`` helper is shared across all three variants to
guarantee consistent metric definitions throughout the codebase.
"""

import json
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score, f1_score,
)

import torch

from src.train_multimodal import _build_enhanced_features


def inference_image_only(model, test_loader, device, model_ckpt, mode='test'):
    """Run inference for the image-only model.

    Parameters
    ----------
    model : nn.Module
        Instantiated CNN (state dict is reloaded from checkpoint inside).
    test_loader : DataLoader
        Yields (path, image, label) tuples.
    device : torch.device
    model_ckpt : str
        Directory containing ``best_model.pt``; results are also written here.
    mode : str
        ``'test'`` → output file is ``test_inference.csv``;
        other values → ``results_{mode}.csv``.

    Returns
    -------
    auc : float   AUROC on the provided split.
    """
    model.load_state_dict(torch.load(os.path.join(model_ckpt, 'best_model.pt'),
                                     map_location=device))
    model.to(device)
    model.eval()

    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for path, images, labels in tqdm(iter(test_loader), desc='Inference'):
            images = images.to(device)
            logit = model(images)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            paths.extend(path)

    metrics = _compute_metrics(trues, probs, preds)
    print(f'[{mode}] AUC: {metrics["auc"]:.3f} | Acc: {metrics["acc"]:.3f}')

    results = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results['image_path'] = results['image_path'].str.split('/').str[-1]
    csv_name = 'test_inference.csv' if mode == 'test' else f'results_{mode}.csv'
    results.to_csv(os.path.join(model_ckpt, csv_name), index=False)

    with open(os.path.join(model_ckpt, f'metrics_{mode}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics['auc']


def inference_feature_only(fc_model, X_test, y_test, path_test, device, model_ckpt,
                            batch_size, mode='test'):
    """Run inference for the feature-only MLP.

    Parameters
    ----------
    fc_model : nn.Module
        MLP (state dict is reloaded from checkpoint inside).
    X_test : np.ndarray   shape (N, n_features)
    y_test : np.ndarray   shape (N,)   integer labels
    path_test : array-like of str      image paths for the test samples
    device : torch.device
    model_ckpt : str
    batch_size : int
    mode : str
        ``'test'`` → ``test_inference.csv``; other → ``results_{mode}.csv``.

    Returns
    -------
    auc : float
    results : pd.DataFrame
    """
    fc_model.load_state_dict(
        torch.load(os.path.join(model_ckpt, 'best_model_feature_only.pt'),
                   map_location=device)
    )
    fc_model.to(device)
    fc_model.eval()

    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            end = min(i + batch_size, len(X_test))
            batch_X = torch.tensor(X_test[i:end], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_test[i:end], dtype=torch.long).to(device)

            logit = fc_model(batch_X)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += [path_test[j] for j in range(i, end)]

    metrics = _compute_metrics(trues, probs, preds)
    print(f'[{mode}] AUC: {metrics["auc"]:.3f} | Acc: {metrics["acc"]:.3f}')

    results = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results['image_path'] = [p.split('/')[-1] for p in results['image_path']]
    csv_name = 'test_inference.csv' if mode == 'test' else f'results_{mode}.csv'
    results.to_csv(os.path.join(model_ckpt, csv_name), index=False)

    with open(os.path.join(model_ckpt, f'metrics_{mode}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics['auc'], results


def inference_multimodal(fc_model, test_features, test_paths, test_df,
                         device, model_ckpt, batch_size, mode='test'):
    """Run inference for the multimodal model.

    Internally calls ``_build_enhanced_features`` to concatenate the
    image feature vectors with the quantitative measurements before
    forwarding through the MLP head.

    Parameters
    ----------
    fc_model : nn.Module
        MLP (state dict is reloaded from checkpoint inside).
    test_features : torch.Tensor   shape (N, 512)
    test_paths : list of str       absolute image paths (same order as features)
    test_df : pd.DataFrame         must contain ``label``, ``img_dir``,
                                   and quantitative measurement columns.
    device : torch.device
    model_ckpt : str
    batch_size : int
    mode : str

    Returns
    -------
    auc : float
    results : pd.DataFrame
    """
    fc_model.load_state_dict(
        torch.load(os.path.join(model_ckpt, 'best_model_multimodal.pt'),
                   map_location=device)
    )
    fc_model.to(device)
    fc_model.eval()

    test_labels = test_df['label'].values
    enhanced_test = _build_enhanced_features(test_features, test_paths, test_df)

    probs, preds, trues, paths = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(enhanced_test), batch_size), desc='Inference'):
            end = min(i + batch_size, len(enhanced_test))
            batch_X = torch.tensor(enhanced_test[i:end], dtype=torch.float32).to(device)
            batch_y = torch.tensor(test_labels[i:end], dtype=torch.long).to(device)

            logit = fc_model(batch_X)
            probability = torch.softmax(logit, dim=1)[:, 1]

            probs += probability.detach().cpu().numpy().tolist()
            preds += logit.argmax(dim=1).detach().cpu().numpy().tolist()
            trues += batch_y.detach().cpu().numpy().tolist()
            paths += [test_paths[j] for j in range(i, end)]

    metrics = _compute_metrics(trues, probs, preds)
    print(f'[{mode}] AUC: {metrics["auc"]:.3f} | Acc: {metrics["acc"]:.3f}')

    results = pd.DataFrame({'image_path': paths, 'prob': probs, 'pred': preds, 'label': trues})
    results['image_path'] = [p.split('/')[-1] for p in results['image_path']]
    csv_name = 'test_inference.csv' if mode == 'test' else f'results_{mode}.csv'
    results.to_csv(os.path.join(model_ckpt, csv_name), index=False)

    with open(os.path.join(model_ckpt, f'metrics_{mode}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics['auc'], results


def _compute_metrics(trues, probs, preds):
    """Compute a full set of binary classification metrics.

    Parameters
    ----------
    trues : list of int   ground-truth labels (0 / 1)
    probs : list of float predicted probabilities for the positive class
    preds : list of int   hard predictions (argmax of logits)

    Returns
    -------
    dict with keys: auc, prauc, acc, sensitivity, specificity, precision, f1
    """
    return {
        'auc':         roc_auc_score(trues, probs),
        'prauc':       average_precision_score(trues, probs),
        'acc':         accuracy_score(trues, preds),
        'sensitivity': recall_score(trues, preds, pos_label=1, zero_division=0),
        'specificity': recall_score(trues, preds, pos_label=0, zero_division=0),
        'precision':   precision_score(trues, preds, zero_division=0),
        'f1':          f1_score(trues, preds, zero_division=0),
    }
