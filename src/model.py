"""
Model factory for image encoders and the fully-connected classifier head.

``get_model`` returns a torchvision backbone with the classification head
replaced by a two-class linear layer initialised with Kaiming uniform
weights.  ``create_fc_model`` builds a lightweight MLP used by both the
feature-only and multimodal pipelines.
"""

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = ['resnet18', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'inceptionv3']

_WEIGHTS_MAP = {
    'resnet18':        models.ResNet18_Weights.IMAGENET1K_V1,
    'densenet121':     models.DenseNet121_Weights.IMAGENET1K_V1,
    'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    'mobilenet_v2':    models.MobileNet_V2_Weights.IMAGENET1K_V1,
    'inceptionv3':     models.Inception_V3_Weights.IMAGENET1K_V1,
}


def get_model(model_name, num_classes, pt=True):
    """Return a torchvision backbone with a re-initialised classification head.

    Parameters
    ----------
    model_name : str
        One of ``SUPPORTED_MODELS``.
    num_classes : int
        Number of output logits (2 for binary classification).
    pt : bool
        Load ImageNet-pretrained weights when ``True``.

    Returns
    -------
    model : nn.Module
    img_size : int
        Expected spatial input dimension (224 or 299 for InceptionV3).
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {SUPPORTED_MODELS}")

    weights = _WEIGHTS_MAP[model_name] if pt else None

    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        new_head = nn.Linear(model.fc.in_features, num_classes)
        _init_head(new_head)
        model.fc = new_head
        img_size = 224

    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        new_head = nn.Linear(model.classifier.in_features, num_classes)
        _init_head(new_head)
        model.classifier = new_head
        img_size = 224

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        new_head = nn.Linear(model.classifier[1].in_features, num_classes)
        _init_head(new_head)
        model.classifier[1] = new_head
        img_size = 224

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        new_head = nn.Linear(model.classifier[1].in_features, num_classes)
        _init_head(new_head)
        model.classifier[1] = new_head
        img_size = 224

    elif model_name == 'inceptionv3':
        # aux_logits=False simplifies the forward pass to a single tensor output.
        model = models.inception_v3(weights=weights, aux_logits=False)
        new_head = nn.Linear(model.fc.in_features, num_classes)
        _init_head(new_head)
        model.fc = new_head
        img_size = 299

    return model, img_size


def _init_head(layer):
    """Kaiming-uniform init for the classification head (bias zeroed)."""
    nn.init.kaiming_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def create_fc_model(input_size, n_layers=2, first_hidden=128, hidden_sizes=None,
                    use_dropout=False, dropout_rate=0.3):
    """Build a variable-depth fully-connected classifier.

    Architecture: Linear(input → first_hidden) → ReLU [→ Dropout]
                  [→ Linear(first_hidden → hidden_sizes[i]) → ReLU [→ Dropout]] ...
                  → Linear(last_hidden → 2)

    Parameters
    ----------
    input_size : int
        Dimensionality of the input feature vector.
    n_layers : int
        Total number of linear layers including the output layer.
        If 1, returns a single linear projection to 2 classes.
    first_hidden : int
        Width of the first hidden layer.
    hidden_sizes : list of int or None
        Widths of additional intermediate hidden layers (between the first
        hidden layer and the output layer).  ``None`` means no intermediate
        layers.
    use_dropout : bool
        Insert a Dropout layer after each hidden activation when ``True``.
    dropout_rate : float
        Dropout probability (used only when ``use_dropout`` is ``True``).
    """
    if hidden_sizes is None:
        hidden_sizes = []

    if n_layers == 1:
        return nn.Linear(input_size, 2)

    layers = [nn.Linear(input_size, first_hidden), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(dropout_rate))

    current_size = first_hidden
    for size in hidden_sizes:
        layers.extend([nn.Linear(current_size, size), nn.ReLU()])
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        current_size = size

    layers.append(nn.Linear(current_size, 2))
    return nn.Sequential(*layers)
