from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from preprocessing.data_loader import load_spt_data


LOADER_PARAMS = {
    "data_variant": "snr",
    "response_template": "response_{year}.hdf5",
    "snr_template": "snr_{year}.hdf5",
    "observation_id_key": "Observation ID",
    "feature_names": {"response": "response", "snr": "SNR"},
}


def _write_spt_file(path: Path, timestamps: list[int], channels: dict[str, list[float]]) -> None:
    with h5py.File(path, "w") as source:
        source.create_dataset("Observation ID", data=timestamps)
        for name, values in channels.items():
            source.create_dataset(name, data=values)


def test_load_spt_snr_preserves_source_order_and_metadata(tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "snr_2019.hdf5", [1, 2], {"2019.fwb": [3, 4], "2019.fwc": [5, 6]})
    _write_spt_file(tmp_path / "snr_2020.hdf5", [3, 4], {"2019.fwb": [7, 8], "2019.fwc": [9, 10]})

    data, metadata = load_spt_data(root=tmp_path, years=(2019, 2020), **LOADER_PARAMS)

    assert data.dtype == np.float32
    assert data.shape == (4, 2, 1)
    np.testing.assert_allclose(data[:, :, 0], [[3, 5], [4, 6], [7, 9], [8, 10]])
    np.testing.assert_array_equal(metadata["timestamps"], [1, 2, 3, 4])
    np.testing.assert_array_equal(metadata["channel_names"], ["2019.fwb", "2019.fwc"])
    np.testing.assert_array_equal(metadata["feature_names"], ["SNR"])
    np.testing.assert_array_equal(metadata["years"], [2019, 2020])
    np.testing.assert_array_equal(metadata["sample_years"], [2019, 2019, 2020, 2020])
    assert metadata["observation_id_key"].item() == "Observation ID"


def test_load_spt_snr_optionally_loads_response_quality_reference(tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "snr_2019.hdf5", [1, 2], {"2019.fwb": [3, 4]})
    _write_spt_file(tmp_path / "response_2019.hdf5", [1, 2], {"2019.fwb": [10, 11]})

    _data, metadata = load_spt_data(
        root=tmp_path,
        years=(2019,),
        load_response_reference=True,
        **LOADER_PARAMS,
    )

    np.testing.assert_allclose(metadata["quality_reference_response"][:, :, 0], [[10], [11]])


def test_load_spt_uses_configured_environment_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "snr_2019.hdf5", [1], {"2019.fwb": [3]})
    monkeypatch.setenv("SPT_DATA_DIR_BENCHMARK", str(tmp_path))

    data, _metadata = load_spt_data(root=None, years=(2019,), **LOADER_PARAMS)

    np.testing.assert_allclose(data[:, :, 0], [[3]])


def test_load_spt_response_selects_response_template(tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "response_2019.hdf5", [1], {"2019.fwb": [1.5]})

    data, metadata = load_spt_data(
        root=tmp_path,
        years=(2019,),
        data_variant="response",
        **{key: value for key, value in LOADER_PARAMS.items() if key != "data_variant"},
    )

    np.testing.assert_allclose(data[:, :, 0], [[1.5]])
    np.testing.assert_array_equal(metadata["feature_names"], ["response"])


def test_load_spt_rejects_non_monotonic_timestamps_when_requested(tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "snr_2019.hdf5", [2, 1], {"2019.fwb": [3, 4]})

    with pytest.raises(ValueError, match="non-monotonic"):
        load_spt_data(root=tmp_path, years=(2019,), **LOADER_PARAMS)


def test_load_spt_allows_non_monotonic_timestamps_when_disabled(tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "snr_2019.hdf5", [2, 1], {"2019.fwb": [3, 4]})

    data, metadata = load_spt_data(
        root=tmp_path,
        years=(2019,),
        require_monotonic_timestamps=False,
        **LOADER_PARAMS,
    )

    np.testing.assert_array_equal(metadata["timestamps"], [2, 1])
    np.testing.assert_allclose(data[:, :, 0], [[3], [4]])


def test_load_spt_real_snr_data() -> None:
    root = os.environ.get("SPT_DATA_DIR_BENCHMARK")
    if not root:
        pytest.skip("SPT_DATA_DIR_BENCHMARK is not configured")

    data, metadata = load_spt_data(
        root=None,
        years=(2019,),
        data_variant="snr",
        response_template="calibrator_responses_095ghz_{year}.hdf5",
        snr_template="calibrator_response_snrs_095ghz_{year}.hdf5",
        observation_id_key="Observation ID",
        feature_names={"response": "response", "snr": "SNR"},
    )

    assert data.dtype == np.float32
    assert data.ndim == 3
    assert data.shape[0] == metadata["timestamps"].shape[0]
    assert data.shape[1] == metadata["channel_names"].shape[0]
    assert data.shape[2] == 1


def test_load_spt_rejects_channel_schema_mismatch(tmp_path: Path) -> None:
    _write_spt_file(tmp_path / "snr_2019.hdf5", [1], {"2019.fwb": [3], "2019.fwc": [4]})
    _write_spt_file(tmp_path / "snr_2020.hdf5", [2], {"2019.fwb": [5], "2019.fwd": [6]})

    with pytest.raises(ValueError, match="Channel schema"):
        load_spt_data(root=tmp_path, years=(2019, 2020), **LOADER_PARAMS)
