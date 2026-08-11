"""
Reducer transforms: compress or aggregate along an axis.

All functions follow the (T, C, F) -> (T, C, F) contract (shape may shrink).
Registered under step type ``reducer``.
"""

from __future__ import annotations

import numpy as np


def subsample_time(data: np.ndarray, *, stride: int) -> np.ndarray:
    """Keep every *stride*-th time step.

    Args:
        data:   Array of shape (T, C, F).
        stride: Step size along the time axis. Must be >= 1.

    Returns:
        Subsampled array. Shape: (T // stride, C, D) approximately.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return data[::stride, :, :]
