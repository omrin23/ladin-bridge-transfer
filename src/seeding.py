"""Centralized seeding for full reproducibility across all libraries."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set seed for Python random, NumPy, PyTorch (CPU + CUDA), and Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed)
    except ImportError:
        pass

    logger.info("All seeds set to %d", seed)
