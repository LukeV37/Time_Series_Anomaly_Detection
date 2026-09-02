"""
Filter transforms: reduce the array along T or C by dropping rows/columns.

All functions follow the (T, C, F) -> (T, C, F) contract.
Registered under step type ``filter``.
"""

from __future__ import annotations

import numpy as np


def _select_channels(metadata: dict[str, object] | None, keep: np.ndarray) -> None:
    if metadata is None:
        return
    if "channel_names" in metadata:
        metadata["channel_names"] = np.asarray(metadata["channel_names"])[keep]


def _select_timesteps(metadata: dict[str, object] | None, keep: np.ndarray | slice) -> None:
    if metadata is None:
        return
    for key in ("timestamps", "sample_years"):
        if key in metadata:
            metadata[key] = np.asarray(metadata[key])[keep]


def drop_nan_channels(
    data: np.ndarray, *, threshold: float = 0.2, metadata: dict[str, object] | None = None
) -> np.ndarray:
    """Drop channels whose NaN fraction exceeds *threshold*."""
    nan_frac = np.isnan(data).mean(axis=(0, 2))
    keep = nan_frac <= threshold
    _select_channels(metadata, keep)
    return data[:, keep, :]


def drop_nan_timesteps(
    data: np.ndarray, *, threshold: float = 0.005, metadata: dict[str, object] | None = None
) -> np.ndarray:
    """Drop time steps whose NaN fraction across channels exceeds *threshold*."""
    nan_frac = np.isnan(data).mean(axis=(1, 2))
    keep = nan_frac <= threshold
    _select_timesteps(metadata, keep)
    return data[keep, :, :]


def _stable_channel_mask(
    reference: np.ndarray,
    *,
    low_quantile: float,
    high_quantile: float,
    tolerance: float,
) -> np.ndarray:
    median = np.nanmedian(reference, axis=(0, 2))
    low, high = np.nanpercentile(reference, [low_quantile, high_quantile], axis=(0, 2))
    keep = (
        np.isfinite(median)
        & (median > 0.0)
        & (low > (1.0 - tolerance) * median)
        & (high < (1.0 + tolerance) * median)
    )
    if not np.any(keep):
        raise ValueError("Quality selection removed every channel.")
    return keep


def select_stable_channels(
    data: np.ndarray,
    *,
    low_quantile: float = 10.0,
    high_quantile: float = 90.0,
    tolerance: float = 0.10,
    reference_metadata_key: str | None = None,
    metadata: dict[str, object] | None = None,
) -> np.ndarray:
    """Keep channels whose percentile range stays within a fraction of their median."""
    reference = data
    if reference_metadata_key:
        if metadata is None or reference_metadata_key not in metadata:
            raise ValueError(f"Missing quality-selection reference {reference_metadata_key!r}.")
        reference = np.asarray(metadata[reference_metadata_key])
    if reference.shape != data.shape:
        raise ValueError(f"Quality-selection reference shape {reference.shape} does not match {data.shape}.")

    keep = _stable_channel_mask(
        reference,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        tolerance=tolerance,
    )
    _select_channels(metadata, keep)
    if metadata is not None and reference_metadata_key:
        metadata[reference_metadata_key] = reference[:, keep, :]
    return data[:, keep, :]


def keep_channel_mask(
    data: np.ndarray,
    *,
    keep: np.ndarray | list[bool],
    metadata: dict[str, object] | None = None,
) -> np.ndarray:
    """Select channels by a train-derived boolean mask."""
    keep_array = np.asarray(keep, dtype=bool)
    if keep_array.ndim != 1:
        raise ValueError(f"Channel mask must be one-dimensional, got shape {keep_array.shape}.")
    if keep_array.shape[0] != data.shape[1]:
        raise ValueError(
            f"Channel mask length {keep_array.shape[0]} does not match channel count {data.shape[1]}."
        )
    if not np.any(keep_array):
        raise ValueError("Channel mask removed every channel.")

    indices = np.flatnonzero(keep_array)
    _select_channels(metadata, keep_array)
    if metadata is not None:
        for key, value in tuple(metadata.items()):
            if key.startswith("quality_reference_"):
                metadata[key] = np.asarray(value)[:, indices, :]
    return data[:, indices, :]


def keep_named_channels(
    data: np.ndarray,
    *,
    channel_names: np.ndarray | list[str],
    metadata: dict[str, object] | None = None,
) -> np.ndarray:
    """Select channels by name while preserving the requested train-derived order."""
    if metadata is None or "channel_names" not in metadata:
        raise ValueError("Named channel selection requires metadata['channel_names'].")

    current_names = np.asarray(metadata["channel_names"], dtype=str)
    requested_names = np.asarray(channel_names, dtype=str)
    index_by_name = {name: index for index, name in enumerate(current_names.tolist())}
    missing = [name for name in requested_names.tolist() if name not in index_by_name]
    if missing:
        raise ValueError(f"Requested channels are missing from the current data: {missing}")

    indices = np.asarray([index_by_name[name] for name in requested_names.tolist()], dtype=np.int64)
    keep = np.zeros(current_names.shape[0], dtype=bool)
    keep[indices] = True
    result = data[:, indices, :]
    _select_channels(metadata, keep)
    if metadata is not None:
        for key, value in tuple(metadata.items()):
            if key.startswith("quality_reference_"):
                metadata[key] = np.asarray(value)[:, indices, :]
        metadata["channel_names"] = requested_names
    return result


def filter_quality_timesteps(
    data: np.ndarray,
    *,
    require_positive: bool = True,
    low_quantile: float = 1.0,
    high_quantile: float = 100.0,
    min_valid_fraction: float = 1.0,
    split: str | None = None,
    trim_test_split: bool = True,
    metadata: dict[str, object] | None = None,
) -> np.ndarray:
    """Keep timesteps finite and within configured per-channel value percentiles."""
    if split == "test" and not trim_test_split:
        return data
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError(f"min_valid_fraction must be in (0, 1], got {min_valid_fraction}.")

    valid = np.isfinite(data)
    if require_positive:
        valid &= data > 0.0

    if low_quantile > 0.0 or high_quantile < 100.0:
        bounds_data = np.where(valid, data, np.nan)
        low, high = np.nanpercentile(bounds_data, [low_quantile, high_quantile], axis=0)
        if low_quantile > 0.0:
            valid &= data > low[np.newaxis, :, :]
        if high_quantile < 100.0:
            valid &= data < high[np.newaxis, :, :]

    keep = np.mean(valid, axis=(1, 2)) >= min_valid_fraction
    if not np.any(keep):
        raise ValueError("Quality selection removed every timestep.")
    _select_timesteps(metadata, keep)
    if metadata is not None:
        for key, value in tuple(metadata.items()):
            if key.startswith("quality_reference_"):
                metadata[key] = np.asarray(value)[keep, :, :]
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
        data:         Array of shape (T, C, F).
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
    selection = slice(start, end)
    _select_timesteps(metadata, selection)
    return data[selection, :, :]
