"""
Reducer transforms: compress or aggregate along an axis.

All functions follow the (T, C, F) -> (T, C, F) contract (shape may shrink).
Registered under step type ``reducer``.
"""

from __future__ import annotations

import numpy as np


def subsample_time(
    data: np.ndarray, *, stride: int, metadata: dict[str, object] | None = None
) -> np.ndarray:
    """Keep every *stride*-th time step."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    selection = slice(None, None, stride)
    if metadata is not None:
        for key in ("timestamps", "sample_years"):
            if key in metadata:
                metadata[key] = np.asarray(metadata[key])[selection]
    return data[selection, :, :]
