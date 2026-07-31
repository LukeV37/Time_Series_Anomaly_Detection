"""
Normalizer transforms: scale or shift values.

All functions follow the (T, C, D) -> (T, C, D) contract.
Registered under step type ``normalizer``.

Note: These are stateless transforms suitable for applying pre-computed
statistics at runtime. This package does not currently implement a separate
fit/transform lifecycle for learned preprocessing state.
"""

from __future__ import annotations

import numpy as np


def clip_values(
    data: np.ndarray,
    *,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    """Clip array values to [low, high].

    Args:
        data: Array of shape (T, C, D).
        low:  Minimum value. None means no lower bound.
        high: Maximum value. None means no upper bound.

    Returns:
        Clipped array. Same shape as input.
    """
    return np.clip(data, low, high)


def subtract_mean(data: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Subtract the mean along *axis* (stateless, computed on the input array).

    Args:
        data: Array of shape (T, C, D).
        axis: Axis along which to compute and subtract the mean (default: 0 = time).

    Returns:
        Mean-centred array. Same shape as input.
    """
    return data - data.mean(axis=axis, keepdims=True)


def apply_scale(
    data: np.ndarray,
    *,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Standardise using pre-computed per-channel mean and std.

    Intended for applying statistics that were fitted on a training split and
    saved separately by surrounding training code.

    Args:
        data: Array of shape (T, C, D).
        mean: Per-channel mean values. Length must equal C.
        std:  Per-channel std values. Length must equal C.

    Returns:
        Standardised array. Same shape as input.
    """
    mu = np.array(mean, dtype=data.dtype).reshape(1, -1, 1)
    sigma = np.array(std, dtype=data.dtype).reshape(1, -1, 1)
    return (data - mu) / np.where(sigma == 0, 1.0, sigma)
