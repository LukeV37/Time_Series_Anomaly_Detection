"""Minimal .npz loading and windowed DataLoader utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _require_3d_data(name: str, data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected {name} shape (T, C, D), got {array.shape}")
    return array


def load_npz_data(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], bool]:
    """Load preprocessing output and return split arrays, metadata, and split provenance."""
    with np.load(Path(path), allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}

    if {"train_data", "val_data", "test_data"}.issubset(arrays):
        splits = {
            "train": _require_3d_data("train_data", arrays.pop("train_data")),
            "val": _require_3d_data("val_data", arrays.pop("val_data")),
            "test": _require_3d_data("test_data", arrays.pop("test_data")),
        }
        return splits, arrays, True

    if "data" not in arrays:
        raise KeyError(f"Expected split-aware keys or 'data' in {path}")

    data = _require_3d_data("data", arrays.pop("data"))
    return {"data": data}, arrays, False


def split_time_series(
    data: np.ndarray,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a time series chronologically into train/val/test arrays."""
    if data.ndim != 3:
        raise ValueError(f"Expected (T, C, D), got {data.shape}")
    n_time = data.shape[0]
    train_end = int(n_time * train_ratio)
    val_end = train_end + int(n_time * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def standardize_splits(
    train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize all splits with per-feature statistics fitted on training data."""
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train_data - mean) / std, (val_data - mean) / std, (test_data - mean) / std


def robust_scale_splits(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust-scale all splits with train-only median and IQR."""
    median = np.median(train_data, axis=0, keepdims=True)
    q25 = np.percentile(train_data, 25.0, axis=0, keepdims=True)
    q75 = np.percentile(train_data, 75.0, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return (
        (train_data - median) / iqr,
        (val_data - median) / iqr,
        (test_data - median) / iqr,
    )


def window_time_series(data: np.ndarray, window_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``(T, C, D)`` data into TranAD windows ``(N, W, F)`` and targets ``(N, 1, F)``."""
    if data.ndim != 3:
        raise ValueError(f"Expected (T, C, D), got {data.shape}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    flat = torch.from_numpy(np.asarray(data, dtype=np.float32)).reshape(data.shape[0], -1)
    if flat.shape[0] < window_size:
        raise ValueError(
            f"Need at least window_size={window_size} timesteps, got {flat.shape[0]}"
        )

    windows = []
    targets = []
    for end in range(window_size - 1, flat.shape[0]):
        start = end - window_size + 1
        windows.append(flat[start : end + 1])
        targets.append(flat[end : end + 1])
    return torch.stack(windows), torch.stack(targets)


def _scale_datasets(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    *,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaling == "standard":
        return standardize_splits(train_data, val_data, test_data)
    if scaling == "robust":
        return robust_scale_splits(train_data, val_data, test_data)
    raise ValueError(f"Unsupported scaling {scaling!r}. Expected 'standard' or 'robust'.")


def create_data_loaders(
    train_npz_path: str | Path,
    *,
    test_npz_path: str | Path | None = None,
    window_size: int,
    batch_size: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    scaling: str = "standard",
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Build train/validation/test DataLoaders from saved preprocessing ``.npz`` files."""
    train_arrays, train_metadata, used_precomputed_split = load_npz_data(train_npz_path)
    if used_precomputed_split:
        train_data = train_arrays["train"]
        val_data = train_arrays["val"]
        train_test_data = train_arrays["test"]
        metadata = dict(train_metadata)
        metadata["split_source"] = np.asarray("precomputed", dtype="<U11")
        test_data = train_test_data
        if test_npz_path is not None:
            test_arrays, test_metadata, test_precomputed = load_npz_data(test_npz_path)
            if test_precomputed:
                raise ValueError("Expected standalone test preprocessing artifact, not precomputed splits.")
            test_data = test_arrays["data"]
            for key in ("timestamps", "channel_names", "feature_names", "years", "sample_years"):
                if key in test_metadata:
                    metadata[f"test_{key}"] = test_metadata[key]
    else:
        train_data, val_data, _discarded_train_tail = split_time_series(
            train_arrays["data"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        metadata = dict(train_metadata)
        metadata["split_source"] = np.asarray("config", dtype="<U6")
        if test_npz_path is None:
            raise ValueError("test_npz_path is required when training from a single train preprocessing artifact.")
        test_arrays, test_metadata, test_precomputed = load_npz_data(test_npz_path)
        if test_precomputed:
            raise ValueError("Expected standalone test preprocessing artifact, not precomputed splits.")
        test_data = test_arrays["data"]
        for key in ("timestamps", "channel_names", "feature_names", "years", "sample_years"):
            if key in test_metadata:
                metadata[f"test_{key}"] = test_metadata[key]

    train_data, val_data, test_data = _scale_datasets(
        train_data,
        val_data,
        test_data,
        scaling=scaling,
    )

    train_windows, train_targets = window_time_series(train_data, window_size)
    val_windows, val_targets = window_time_series(val_data, window_size)
    test_windows, test_targets = window_time_series(test_data, window_size)

    train_loader = DataLoader(
        TensorDataset(train_windows, train_targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TensorDataset(val_windows, val_targets),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(test_windows, test_targets),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, metadata
