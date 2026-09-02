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


def interpolate_nan_per_channel(
    data: np.ndarray, *, metadata: dict[str, object] | None = None
) -> np.ndarray:
    """Linearly interpolate NaNs per channel using loader timestamps."""
    if metadata is None or "timestamps" not in metadata:
        raise ValueError("interpolate_nan_per_channel requires metadata['timestamps'].")
    if not np.isnan(data).any():
        return data

    timestamps = np.asarray(metadata["timestamps"], dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.shape[0] != data.shape[0]:
        raise ValueError(
            "interpolate_nan_per_channel requires one timestamp per timestep. "
            f"Got timestamps={timestamps.shape}, data={data.shape}."
        )

    result = np.array(data, copy=True)
    for channel in range(result.shape[1]):
        for feature in range(result.shape[2]):
            values = result[:, channel, feature]
            nan_mask = np.isnan(values)
            if not nan_mask.any():
                continue
            good_mask = ~nan_mask
            if good_mask.sum() < 2:
                continue
            values[nan_mask] = np.interp(timestamps[nan_mask], timestamps[good_mask], values[good_mask])
    return result


def drop_features(
    data: np.ndarray, *, indices: list[int], metadata: dict[str, object] | None = None
) -> np.ndarray:
    """Drop specific feature (F-axis) indices."""
    keep = np.ones(data.shape[2], dtype=bool)
    keep[indices] = False
    if metadata is not None and "feature_names" in metadata:
        metadata["feature_names"] = np.asarray(metadata["feature_names"])[keep]
    return data[:, :, keep]


def keep_features(
    data: np.ndarray, *, indices: list[int], metadata: dict[str, object] | None = None
) -> np.ndarray:
    """Keep only the specified feature (F-axis) indices."""
    if metadata is not None and "feature_names" in metadata:
        metadata["feature_names"] = np.asarray(metadata["feature_names"])[indices]
    return data[:, :, indices]
