"""
Global configuration: reproducibility helpers and device selection.

Call seed_everything(seed) once at the start of each fold to ensure
deterministic behaviour across NumPy, PyTorch CPU/CUDA, and Python's
random module.  seed_worker is passed to DataLoader as worker_init_fn
so that each worker process also receives a deterministic seed derived
from the global torch seed.
"""

import random
import os
import numpy as np
import torch


def seed_everything(seed):
    """Fix all sources of randomness for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable cuDNN auto-tuner; force deterministic CUDA kernels.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Suppress non-determinism from oneDNN floating-point optimisations.
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def seed_worker(_worker_id):
    """Per-worker seed initialiser for DataLoader reproducibility.

    PyTorch derives each worker's initial seed from the main process seed
    plus the worker id.  This function propagates that seed to NumPy and
    Python's random module so all worker-side operations are deterministic.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
