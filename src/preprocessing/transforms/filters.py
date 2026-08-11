"""
Filter transforms: reduce the array along T or C by dropping rows/columns.

All functions follow the (T, C, D) -> (T, C, D) contract.
Registered under step type ``filter``.
"""

from __future__ import annotations

import numpy as np


def drop_nan_channels(data: np.ndarray, *, threshold: float = 0.2) -> np.ndarray:
    """Drop channels whose NaN fraction exceeds *threshold*.

    Args:
        data:      Array of shape (T, C, D).
        threshold: Maximum allowed NaN fraction per channel (0.0 – 1.0).

    Returns:
        Array with offending channels removed. Shape: (T, C', D).
    """
    # nan fraction per channel: mean over T and D axes
    nan_frac = np.isnan(data).mean(axis=(0, 2))  # (C,)
    keep = nan_frac <= threshold
    return data[:, keep, :]


def drop_nan_timesteps(data: np.ndarray, *, threshold: float = 0.005) -> np.ndarray:
    """Drop time steps whose NaN fraction across channels exceeds *threshold*.

    Args:
        data:      Array of shape (T, C, D).
        threshold: Maximum allowed NaN fraction per time step (0.0 – 1.0).

    Returns:
        Array with offending time steps removed. Shape: (T', C, D).
    """
    nan_frac = np.isnan(data).mean(axis=(1, 2))  # (T,)
    keep = nan_frac <= threshold
    return data[keep, :, :]


def trim_edges(
    data: np.ndarray,
    *,
    remove_first: int = 0,
    remove_last: int = 0,
    metadata: dict[str, object] | None = None,
    run_specific: dict[str, dict[str, int]] | None = None,
) -> np.ndarray:
    """Remove a fixed number of time steps from the start and/or end.

    Useful for stripping initialisation artefacts or run-end noise.
    If ``run_specific`` is provided, a matching ``metadata['run_number']`` can
    override the default trim values for the current input only.

    Args:
        data:         Array of shape (T, C, D).
        remove_first: Number of leading time steps to drop.
        remove_last:  Number of trailing time steps to drop.
        metadata:     Optional loader metadata for the current input.
        run_specific: Optional per-run trim map, keyed by run number.

    Returns:
        Trimmed array. Shape: (T - remove_first - remove_last, C, D).
    """
    run_number = None if metadata is None else metadata.get("run_number")
    if run_specific and run_number is not None:
        overrides = run_specific.get(str(run_number))
        if overrides is not None:
            remove_first = overrides.get("remove_first", remove_first)
            remove_last = overrides.get("remove_last", remove_last)

    t = data.shape[0]
    start = remove_first
    end = t - remove_last if remove_last > 0 else t
    return data[start:end, :, :]
