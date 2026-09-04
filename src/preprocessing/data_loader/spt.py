"""Minimal SPT HDF5 loader for canonical ``(T, C, F)`` preprocessing input."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def load_spt_data(
    *,
    root: str | Path | None = None,
    years: tuple[int, ...],
    data_variant: str,
    response_template: str,
    snr_template: str,
    observation_id_key: str,
    feature_names: dict[str, str],
    require_monotonic_timestamps: bool = True,
    load_response_reference: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one configured SPT HDF5 variant without altering its samples or channels."""
    if not years:
        raise ValueError("SPT loader requires at least one configured year.")
    if data_variant not in {"response", "snr"}:
        raise ValueError("SPT data_variant must be 'response' or 'snr'.")
    if data_variant not in feature_names:
        raise ValueError(f"SPT loader requires feature_names[{data_variant!r}].")

    template = response_template if data_variant == "response" else snr_template
    if not template:
        raise ValueError(f"SPT loader requires a template for {data_variant!r} data.")

    root_value = root or os.environ.get("SPT_DATA_DIR_BENCHMARK")
    if not root_value:
        raise ValueError("SPT loader requires root or SPT_DATA_DIR_BENCHMARK.")
    root_path = Path(root_value).expanduser().resolve()
    values_by_year: list[np.ndarray] = []
    timestamps_by_year: list[np.ndarray] = []
    sample_years: list[np.ndarray] = []
    data_paths: list[str] = []
    response_values_by_year: list[np.ndarray] = []
    channel_names: np.ndarray | None = None

    for year in years:
        path = root_path / template.format(year=year)
        if not path.is_file():
            raise FileNotFoundError(f"Configured SPT input file does not exist: {path}")

        with h5py.File(path, "r") as source:
            if observation_id_key not in source:
                raise KeyError(f"{path} does not contain timestamp key {observation_id_key!r}")

            timestamps = np.asarray(source[observation_id_key][:])
            if timestamps.ndim != 1:
                raise ValueError(
                    f"{path}:{observation_id_key} must be one-dimensional, got {timestamps.shape}."
                )
            if require_monotonic_timestamps and np.any(timestamps[1:] < timestamps[:-1]):
                raise ValueError(f"{path} has non-monotonic timestamps.")

            current_channel_names = np.asarray(
                [key for key in source.keys() if key != observation_id_key], dtype=str
            )
            if current_channel_names.size == 0:
                raise ValueError(f"{path} contains no detector datasets.")

            channels = []
            for name in current_channel_names:
                values = np.asarray(source[name][:])
                if values.ndim != 1:
                    raise ValueError(f"{path}:{name} must be one-dimensional, got {values.shape}.")
                if values.shape[0] != timestamps.shape[0]:
                    raise ValueError(
                        f"{path}:{name} has {values.shape[0]} samples; "
                        f"{observation_id_key!r} has {timestamps.shape[0]}."
                    )
                channels.append(values)

        if channel_names is None:
            channel_names = current_channel_names
        elif not np.array_equal(current_channel_names, channel_names):
            raise ValueError(f"Channel schema in {path} does not match the first configured input.")

        values_by_year.append(np.column_stack(channels))
        if data_variant == "snr" and load_response_reference:
            response_path = root_path / response_template.format(year=year)
            if not response_path.is_file():
                raise FileNotFoundError(
                    f"Configured SPT response reference does not exist: {response_path}"
                )
            with h5py.File(response_path, "r") as response_source:
                if observation_id_key not in response_source:
                    raise KeyError(
                        f"{response_path} does not contain timestamp key {observation_id_key!r}"
                    )
                response_timestamps = np.asarray(response_source[observation_id_key][:])
                if not np.array_equal(response_timestamps, timestamps):
                    raise ValueError(
                        f"{response_path} timestamps do not match {path}."
                    )
                if set(response_source.keys()) != set(current_channel_names) | {observation_id_key}:
                    raise ValueError(f"Channel schema in {response_path} does not match {path}.")
                response_values_by_year.append(
                    np.column_stack([np.asarray(response_source[name][:]) for name in current_channel_names])
                )
        timestamps_by_year.append(timestamps)
        sample_years.append(np.full(timestamps.shape[0], year, dtype=np.int64))
        data_paths.append(str(path))

    values = np.concatenate(values_by_year, axis=0)
    timestamps = np.concatenate(timestamps_by_year, axis=0)
    data = values.astype(np.float32, copy=False)[:, :, None]
    metadata = {
        "timestamps": timestamps,
        "channel_names": channel_names,
        "feature_names": np.asarray([feature_names[data_variant]], dtype=str),
        "years": np.asarray(years, dtype=np.int64),
        "sample_years": np.concatenate(sample_years),
        "observation_id_key": np.asarray(observation_id_key, dtype=str),
        "data_variant": np.asarray(data_variant, dtype=str),
        "data_paths": np.asarray(data_paths, dtype=str),
    }
    if response_values_by_year:
        metadata["quality_reference_response"] = (
            np.concatenate(response_values_by_year, axis=0).astype(np.float32, copy=False)[:, :, None]
        )
    return data, metadata
