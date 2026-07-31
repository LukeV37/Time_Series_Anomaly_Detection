"""
Imputer transforms: replace missing values using data-driven estimates.

All functions follow the (T, C, D) -> (T, C, D) contract.
Registered under step type ``imputer``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _fill_channel_stat(data: np.ndarray, stat_fn: Callable) -> np.ndarray:
    """Fill NaNs per (C, D) slice using stat_fn computed over the time axis."""
    stats = stat_fn(data, axis=0)  # (C, D)
    nan_mask = np.isnan(data)
    return np.where(nan_mask, stats[np.newaxis, :, :], data)


def fill_channel_median(data: np.ndarray) -> np.ndarray:
    """Fill NaNs in each channel with that channel's median (over time).

    Args:
        data: Array of shape (T, C, D).

    Returns:
        Array with per-channel NaNs replaced by the channel median.
        Same shape as input.
    """
    return _fill_channel_stat(data, np.nanmedian)


def fill_channel_mean(data: np.ndarray) -> np.ndarray:
    """Fill NaNs in each channel with that channel's mean (over time).

    Args:
        data: Array of shape (T, C, D).

    Returns:
        Array with per-channel NaNs replaced by the channel mean.
        Same shape as input.
    """
    return _fill_channel_stat(data, np.nanmean)
