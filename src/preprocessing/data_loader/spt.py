"""Minimal SPT HDF5 loaders for benchmark calibrator-response data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_BENCHMARK_ROOT = Path(
    "/lcrc/project/SPT3G/users/ac.weiquan/ml_dqm/calibrator_responses"
)
DEFAULT_BOLOPROPERTIES_PATH = Path(
    "/lcrc/project/SPT3G/analysis/calarchive/v3/boloproperties/60000000.g3"
)
DEFAULT_WAFER_ID = "w206"
DEFAULT_OBSERVATION_ID_KEY = "Observation ID"
DEFAULT_RESPONSE_TEMPLATE = "calibrator_responses_095ghz_{year}.hdf5"
DEFAULT_YEARS = (2019, 2020, 2021, 2022, 2023)
DEFAULT_DETECTOR_STABILITY_QUANTILES = (10.0, 90.0)
DEFAULT_DETECTOR_STABILITY_TOLERANCE = 0.10
DEFAULT_TIMESTAMP_VALUE_QUANTILES = (1.0, 99.0)
DEFAULT_REQUIRE_POSITIVE = True


def load_spt_benchmark_hdf5(
    root: str | os.PathLike[str] | None = None,
    *,
    years: tuple[int, ...] = DEFAULT_YEARS,
) -> np.ndarray:
    """Load benchmark SPT calibrator-response HDF5 seasons as ``(T, C, 1)``."""
    data, _ = load_spt_benchmark_hdf5_with_metadata(root=root, years=years)
    return data


def load_spt_benchmark_hdf5_with_metadata(
    root: str | os.PathLike[str] | None = None,
    *,
    years: tuple[int, ...] = DEFAULT_YEARS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load benchmark SPT calibrator-response HDF5 data and metadata.

    Hardcoded defaults intentionally mirror the current reference preprocessing:
    - detector stability trim using 10/90 percentiles within median +/- 10%
    - timestamp trim using per-detector 1/99 percentiles
    - require positive and finite detector values

    Returns:
        Tuple ``(data, metadata)`` where ``data`` has shape ``(T, C, 1)``.
    """
    data_root = _resolve_root(root)
    data_paths = _build_data_paths(data_root, years)
    season_payloads, first_order, common_detectors = _read_hdf5_seasons(data_paths)

    det_names = np.asarray([det for det in first_order if det in common_detectors], dtype=object)
    if det_names.size == 0:
        raise RuntimeError("No common detector keys were found across SPT HDF5 files.")

    det_names = _filter_detectors_by_wafer(det_names)
    obs_data = _stack_season_payloads(season_payloads, det_names)
    timestamps = np.concatenate([ts for _, ts, _ in season_payloads]).astype(np.int64, copy=False)

    obs_data, det_names = _trim_detectors_by_stability(obs_data, det_names)
    obs_data, timestamps = _trim_timestamps(obs_data, timestamps, det_names)

    order = np.argsort(timestamps, kind="stable")
    obs_data = obs_data[order]
    timestamps = timestamps[order]

    data = obs_data.astype(np.float32, copy=False)[:, :, None]
    metadata = {
        "timestamps": timestamps,
        "detector_names": det_names,
        "wafer_id": DEFAULT_WAFER_ID,
        "boloproperties_path": str(DEFAULT_BOLOPROPERTIES_PATH),
        "years": tuple(int(year) for year in years),
        "data_paths": [str(path) for path in data_paths],
        "observation_id_key": DEFAULT_OBSERVATION_ID_KEY,
    }
    return data, metadata


def _resolve_root(root: str | os.PathLike[str] | None) -> Path:
    if root is not None:
        resolved = Path(root)
    else:
        env_root = os.environ.get("SPT_DATA_DIR_BENCHMARK")
        resolved = Path(env_root) if env_root else DEFAULT_BENCHMARK_ROOT
    if not resolved.exists():
        raise FileNotFoundError(f"SPT benchmark root does not exist: {resolved}")
    return resolved


def _build_data_paths(root: Path, years: tuple[int, ...]) -> list[Path]:
    data_paths = [root / DEFAULT_RESPONSE_TEMPLATE.format(year=year) for year in years]
    missing = [str(path) for path in data_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing SPT benchmark HDF5 files: {missing}")
    return data_paths


def _read_hdf5_seasons(
    data_paths: list[Path],
) -> tuple[list[tuple[Path, np.ndarray, dict[str, np.ndarray]]], list[str], set[str]]:
    season_payloads: list[tuple[Path, np.ndarray, dict[str, np.ndarray]]] = []
    first_order: list[str] | None = None
    common_detectors: set[str] | None = None

    for data_path in data_paths:
        with h5py.File(data_path, "r") as fobj:
            if DEFAULT_OBSERVATION_ID_KEY not in fobj:
                raise KeyError(
                    f"{data_path} does not contain timestamp key {DEFAULT_OBSERVATION_ID_KEY!r}"
                )
            timestamps = np.asarray(fobj[DEFAULT_OBSERVATION_ID_KEY][:], dtype=np.int64)
            detectors = [str(key) for key in fobj.keys() if str(key) != DEFAULT_OBSERVATION_ID_KEY]
            if first_order is None:
                first_order = detectors
            detector_set = set(detectors)
            common_detectors = detector_set if common_detectors is None else common_detectors & detector_set
            payload = {det: np.asarray(fobj[det][:], dtype=float) for det in detectors}

        for det, values in payload.items():
            if values.shape[0] != timestamps.shape[0]:
                raise ValueError(
                    f"Detector {det!r} in {data_path} has length {values.shape[0]}, "
                    f"expected {timestamps.shape[0]}."
                )
        season_payloads.append((data_path, timestamps, payload))

    if not season_payloads or first_order is None or common_detectors is None:
        raise RuntimeError("No SPT calibration-response HDF5 data were provided.")
    return season_payloads, first_order, common_detectors


def _stack_season_payloads(
    season_payloads: list[tuple[Path, np.ndarray, dict[str, np.ndarray]]],
    det_names: np.ndarray,
) -> np.ndarray:
    return np.vstack([
        np.column_stack([payload[str(det)] for det in det_names])
        for _, _, payload in season_payloads
    ])


def _filter_detectors_by_wafer(det_names: np.ndarray) -> np.ndarray:
    if not DEFAULT_WAFER_ID:
        return det_names
    if not DEFAULT_BOLOPROPERTIES_PATH.exists():
        raise FileNotFoundError(
            f"SPT boloproperties file does not exist: {DEFAULT_BOLOPROPERTIES_PATH}"
        )

    try:
        from spt3g import core
        try:
            from spt3g import calibration  # noqa: F401
        except Exception:
            calibration = None  # type: ignore[assignment]
    except Exception as exc:
        raise ImportError(
            "SPT wafer filtering requires spt3g to read BolometerProperties."
        ) from exc

    try:
        bpm = core.G3File(str(DEFAULT_BOLOPROPERTIES_PATH)).next()["BolometerProperties"]
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SPT BolometerProperties for wafer filtering. "
            "This runtime likely lacks the required SPT3G calibration type registrations."
        ) from exc

    keep: list[int] = []
    for index, det in enumerate(det_names):
        try:
            wafer = getattr(bpm[str(det)], "wafer_id", None)
        except Exception:
            continue
        if wafer == DEFAULT_WAFER_ID:
            keep.append(index)

    if not keep:
        raise RuntimeError(
            f"No detectors matched wafer_id={DEFAULT_WAFER_ID!r} using "
            f"{DEFAULT_BOLOPROPERTIES_PATH}."
        )
    return det_names[np.asarray(keep, dtype=int)]


def _trim_detectors_by_stability(
    obs_data: np.ndarray,
    det_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qlo, qhi = DEFAULT_DETECTOR_STABILITY_QUANTILES
    med = np.nanmedian(obs_data, axis=0)
    p_lo, p_hi = np.nanpercentile(obs_data, [qlo, qhi], axis=0)
    tol = DEFAULT_DETECTOR_STABILITY_TOLERANCE
    stable = (
        np.isfinite(med)
        & (med > 0.0)
        & (p_lo > (1.0 - tol) * med)
        & (p_hi < (1.0 + tol) * med)
    )
    if not np.any(stable):
        raise RuntimeError("SPT calibration-response detector trimming removed every detector.")
    return obs_data[:, stable], det_names[stable]


def _trim_timestamps(
    obs_data: np.ndarray,
    timestamps: np.ndarray,
    det_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(obs_data)
    if DEFAULT_REQUIRE_POSITIVE:
        good &= obs_data > 0.0

    qlo, qhi = DEFAULT_TIMESTAMP_VALUE_QUANTILES
    lo = np.empty(obs_data.shape[1], dtype=float)
    hi = np.empty(obs_data.shape[1], dtype=float)
    for channel_index in range(obs_data.shape[1]):
        valid = good[:, channel_index]
        if not np.any(valid):
            raise RuntimeError(
                f"SPT detector {det_names[channel_index]!r} has no valid samples."
            )
        lo[channel_index], hi[channel_index] = np.percentile(
            obs_data[valid, channel_index], [qlo, qhi]
        )

    if qlo > 0.0:
        good &= obs_data > lo[None, :]
    if qhi < 100.0:
        good &= obs_data < hi[None, :]

    keep = np.all(good, axis=1)
    if not np.any(keep):
        raise RuntimeError("SPT calibration-response timestamp trimming removed every row.")
    return obs_data[keep], timestamps[keep]
