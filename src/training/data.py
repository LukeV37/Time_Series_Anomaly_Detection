"""Minimal .npz loading and windowed DataLoader utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_npz_data(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load saved preprocessing output and return data plus metadata arrays."""
    with np.load(Path(path), allow_pickle=True) as payload:
        arrays = {key: payload[key] for key in payload.files}
    if "data" not in arrays:
        raise KeyError(f"Expected 'data' in {path}")
    data = np.asarray(arrays.pop("data"), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected data shape (T, C, D), got {data.shape}")
    return data, arrays


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


def create_data_loaders(
    npz_path: str | Path,
    *,
    window_size: int,
    batch_size: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    """Build train/val/test DataLoaders from a saved preprocessing ``.npz`` file."""
    data, metadata = load_npz_data(npz_path)
    train_data, val_data, test_data = split_time_series(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
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
