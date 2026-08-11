"""ATLAS CSV loader for merged run exports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_atlas_data(
    csv_path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    run_number: int | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an ATLAS merged CSV and return data plus metadata.

    The expected input is the merged fetch output where each base signal column may
    have a companion ``*_deltaT`` column. Output features are arranged as
    ``(value, deltaT)`` so the final array shape is ``(T, C, 2)``.
    """
    path = _resolve_csv_path(csv_path=csv_path, root=root, run_number=run_number)
    frame = pd.read_csv(path)

    if "timestamp" not in frame.columns:
        raise ValueError(f"ATLAS CSV is missing required 'timestamp' column: {path}")

    timestamps = frame.pop("timestamp").to_numpy()
    base_columns = [column for column in frame.columns if not column.endswith("_deltaT")]
    if not base_columns:
        raise ValueError(f"ATLAS CSV has no signal columns: {path}")

    data = np.empty((len(frame), len(base_columns), 2), dtype=np.float64)
    missing_delta_t = []
    for index, column in enumerate(base_columns):
        delta_t_column = f"{column}_deltaT"
        if delta_t_column not in frame.columns:
            missing_delta_t.append(delta_t_column)
            continue
        data[:, index, 0] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        data[:, index, 1] = pd.to_numeric(frame[delta_t_column], errors="coerce").to_numpy(dtype=np.float64)

    if missing_delta_t:
        preview = ", ".join(missing_delta_t[:5])
        raise ValueError(
            f"ATLAS CSV is missing expected deltaT columns for {len(missing_delta_t)} channels: {preview}"
        )

    metadata = {
        "timestamps": timestamps,
        "channel_names": np.asarray(base_columns, dtype=object),
        "detector_names": np.asarray(base_columns, dtype=object),
        "feature_names": np.asarray(["value", "deltaT"], dtype=object),
        "source_path": str(path),
    }
    if run_number is not None:
        metadata["run_number"] = str(run_number)

    return data, metadata


def _resolve_csv_path(
    csv_path: str | Path | None,
    *,
    root: str | Path | None,
    run_number: int | str | None,
) -> Path:
    if csv_path is not None:
        return Path(csv_path).expanduser().resolve()

    root_dir = Path(root or os.environ.get("ATLAS_DATA_DIR", "")).expanduser()
    if not root_dir:
        raise ValueError("ATLAS CSV loader requires csv_path, root, or ATLAS_DATA_DIR.")

    if run_number is None:
        raise ValueError("ATLAS CSV loader requires run_number when csv_path is not provided.")

    return (root_dir / str(run_number) / "merged.csv").resolve()
