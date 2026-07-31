"""Minimal adapters between labeled tables and NumPy arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dataframe_to_array(dataframe: pd.DataFrame) -> np.ndarray:
    """Convert a merged ATLAS DataFrame to a `(T, C, 2)` NumPy array.

    Feature 0 is the raw channel value.
    Feature 1 is the existing per-channel ``deltaT`` value from the matching
    ``*_deltaT`` column produced during alignment.

    Columns named ``timestamp`` and columns ending in ``_deltaT`` are not
    treated as channels themselves.

    # ponytail: this drops DataFrame metadata for now, including timestamp
    # values, index values, index name, original column labels, and any attrs.
    # If we need round-tripping or labeled outputs later, return metadata next
    # to the array instead of rebuilding it from conventions.
    """
    value_columns = [
        column
        for column in dataframe.columns
        if column != "timestamp" and not column.endswith("_deltaT")
    ]

    array = np.empty((len(dataframe), len(value_columns), 2), dtype=float)
    for channel_index, column in enumerate(value_columns):
        delta_t_column = f"{column}_deltaT"
        if delta_t_column not in dataframe.columns:
            raise ValueError(f"Missing deltaT column for {column!r}: expected {delta_t_column!r}")
        array[:, channel_index, 0] = dataframe[column].to_numpy(dtype=float, copy=True)
        array[:, channel_index, 1] = dataframe[delta_t_column].to_numpy(dtype=float, copy=True)
    return array
