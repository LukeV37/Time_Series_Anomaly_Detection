"""
Transform steps: reshape or rearrange the array without changing its scale.

All functions follow the (T, C, F) -> (T, C, F) contract.
Registered under step type ``transform``.
"""

from __future__ import annotations

import numpy as np


def fill_nan(data: np.ndarray, *, value: float = 0.0) -> np.ndarray:
    """Replace all NaN entries with a constant value.

    Args:
        data:  Array of shape (T, C, F).
        value: Replacement value (default 0.0).

    Returns:
        Array with NaNs replaced. Same shape as input.
    """
    return np.where(np.isnan(data), value, data)


def drop_features(data: np.ndarray, *, indices: list[int]) -> np.ndarray:
    """Drop specific feature (F-axis) indices.

    Args:
        data:    Array of shape (T, C, F).
        indices: F-axis positions to remove.

    Returns:
        Array with specified features removed. Shape: (T, C, F').
    """
    return np.delete(data, indices, axis=2)


def keep_features(data: np.ndarray, *, indices: list[int]) -> np.ndarray:
    """Keep only the specified feature (F-axis) indices.

    Args:
        data:    Array of shape (T, C, F).
        indices: F-axis positions to retain.

    Returns:
        Array containing only the selected features. Shape: (T, C, F').
    """
    return data[:, :, indices]
