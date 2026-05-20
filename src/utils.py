"""
Utility functions for computing inverse-frequency class weights.

Two variants are provided: one that iterates over a DataLoader (used when
the dataset is already wrapped in a loader) and one that operates directly
on a DataFrame (used for the feature-only pipeline where no image loading
is required).

Weight formula: w_c = 1 - (n_c / N), so the minority class receives a
higher weight.  This is the complement-frequency weighting scheme.
"""

import torch


def calculate_class_weight_from_loader(train_loader, num_classes=2):
    """Compute per-class weights by iterating over a DataLoader.

    Parameters
    ----------
    train_loader : DataLoader
        Must yield (path, image, label) tuples.
    num_classes : int
        Number of distinct class labels (default 2 for binary tasks).

    Returns
    -------
    weights : list of float
        Complement-frequency weight for each class.
    class_weight_tensor : torch.FloatTensor
        Same values as a 1-D tensor, ready to pass to ``nn.CrossEntropyLoss``.
    """
    class_counts = [0] * num_classes
    total = 0
    for _, _, labels in train_loader:
        for label in labels.view(-1):
            class_counts[label.item()] += 1
            total += 1
    weights = [1 - (count / total) for count in class_counts]
    return weights, torch.FloatTensor(weights)


def calculate_class_weight_from_df(train_df, num_classes=2):
    """Compute per-class weights from a DataFrame ``label`` column.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must contain an integer ``label`` column.
    num_classes : int
        Number of distinct class labels (default 2 for binary tasks).

    Returns
    -------
    weights : list of float
    class_weight_tensor : torch.FloatTensor
    """
    class_counts = (
        train_df['label']
        .value_counts()
        .sort_index()
        .reindex(range(num_classes), fill_value=0)
    )
    total = class_counts.sum()
    weights = [1 - (class_counts[i] / total) for i in range(num_classes)]
    return weights, torch.FloatTensor(weights)
