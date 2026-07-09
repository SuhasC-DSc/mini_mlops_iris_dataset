"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility.

    Args:
        seed: Seed value to apply to Python's `random` module and NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)
