from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.transforms.filters import (
    filter_quality_timesteps,
    keep_channel_mask,
    keep_named_channels,
    trim_edges,
)
from preprocessing.transforms.reducer import subsample_time
from preprocessing.transforms.transforms import interpolate_nan_per_channel


def test_pipeline_from_config_file_builds_steps() -> None:
    pipeline = PreprocessingPipeline.from_config_file("configs/spt_pipeline_no_trim.yaml")

    assert repr(pipeline) == "PreprocessingPipeline(steps=['interpolate_nan_per_channel'])"


def test_pipeline_run_applies_steps_in_order() -> None:
    pipeline = PreprocessingPipeline(
        {
            "loader": {"type": "spt", "params": {}},
            "steps": [
                {"name": "fill_nan", "params": {"value": 1.5}},
                {"name": "clip_values", "params": {"low": 0.0, "high": 1.0}},
            ],
        }
    )
    data = np.array(
        [
            [[np.nan, -2.0], [0.5, 3.0]],
            [[-1.0, 2.0], [np.nan, 0.2]],
        ]
    )

    result = pipeline.run(data)

    expected = np.array(
        [
            [[1.0, 0.0], [0.5, 1.0]],
            [[0.0, 1.0], [1.0, 0.2]],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_pipeline_load_and_run_saves_output(tmp_path: Path) -> None:
    pipeline = PreprocessingPipeline(
        {
            "loader": {"type": "spt", "params": {}},
            "output": {
                "save": True,
                "root": str(tmp_path),
                "experiment": "preprocessing",
                "data_tag": "unit",
                "file_name": "result.npz",
            },
            "steps": [{"name": "fill_nan", "params": {"value": 0.0}}],
        }
    )

    loaded = np.array([[[np.nan], [2.0]]])
    metadata = {
        "timestamps": [10],
        "channel_names": ["a", "b"],
        "feature_names": ["SNR"],
        "years": [2019],
        "sample_years": [2019],
    }
    pipeline.load = lambda: (loaded, metadata)  # type: ignore[method-assign]

    result, result_metadata = pipeline.load_and_run()

    np.testing.assert_allclose(result, np.array([[[0.0], [2.0]]]))
    output_path = tmp_path / "preprocessing" / "unit" / "result.npz"
    assert output_path.exists()
    assert result_metadata["output_path"] == str(output_path)
    assert result_metadata["pipeline_config"] == {"steps": pipeline._config["steps"]}

    with np.load(output_path) as saved:
        assert saved.files == [
            "data",
            "timestamps",
            "channel_names",
            "feature_names",
            "years",
            "sample_years",
        ]
        np.testing.assert_allclose(saved["data"], result)
        np.testing.assert_array_equal(saved["timestamps"], np.array([10]))
        np.testing.assert_array_equal(saved["channel_names"], np.array(["a", "b"]))
        np.testing.assert_array_equal(saved["feature_names"], np.array(["SNR"]))
        np.testing.assert_array_equal(saved["years"], np.array([2019]))
        np.testing.assert_array_equal(saved["sample_years"], np.array([2019]))

    metadata_path = tmp_path / "preprocessing" / "unit" / "metadata.json"
    assert not metadata_path.exists()


def test_pipeline_save_persists_run_number(tmp_path: Path) -> None:
    pipeline = PreprocessingPipeline(
        {
            "output": {
                "save": True,
                "root": str(tmp_path),
                "experiment": "preprocessing",
                "data_tag": "unit",
                "file_name": "result.npz",
            },
            "steps": [],
        }
    )

    output_path = pipeline.save_output(np.zeros((2, 1, 1)), {"run_number": "520705"})

    assert output_path is not None
    with np.load(output_path, allow_pickle=True) as saved:
        assert "run_number" in saved.files
        np.testing.assert_array_equal(saved["run_number"], np.array("520705"))


def test_shape_selection_steps_keep_metadata_aligned() -> None:
    pipeline = PreprocessingPipeline(
        {
            "steps": [
                {"name": "drop_nan_channels", "params": {"threshold": 0.34}},
                {"name": "drop_nan_timesteps", "params": {"threshold": 0.0}},
            ]
        }
    )
    data = np.array(
        [
            [[1.0], [np.nan], [10.0]],
            [[np.nan], [np.nan], [11.0]],
            [[3.0], [np.nan], [12.0]],
        ]
    )
    metadata: dict[str, object] = {
        "timestamps": np.array([10, 20, 30]),
        "sample_years": np.array([2019, 2020, 2020]),
        "channel_names": np.array(["a", "b", "c"]),
    }

    result = pipeline.run(data, metadata)

    np.testing.assert_allclose(result[:, :, 0], [[1.0, 10.0], [3.0, 12.0]])
    np.testing.assert_array_equal(metadata["timestamps"], [10, 30])
    np.testing.assert_array_equal(metadata["sample_years"], [2019, 2020])
    np.testing.assert_array_equal(metadata["channel_names"], ["a", "c"])


def test_trim_and_subsample_keep_time_metadata_aligned() -> None:
    data = np.arange(6, dtype=float).reshape(6, 1, 1)
    metadata: dict[str, object] = {
        "timestamps": np.arange(10, 70, 10),
        "sample_years": np.full(6, 2019),
        "run_number": "520705",
    }

    trimmed = trim_edges(
        data,
        remove_first=1,
        remove_last=1,
        metadata=metadata,
        run_specific={"520705": {"remove_first": 2, "remove_last": 0}},
    )
    result = subsample_time(trimmed, stride=2, metadata=metadata)

    np.testing.assert_allclose(result[:, 0, 0], np.array([2.0, 4.0]))
    np.testing.assert_array_equal(metadata["timestamps"], [30, 50])
    np.testing.assert_array_equal(metadata["sample_years"], [2019, 2019])


def test_interpolate_nan_per_channel_uses_timestamps() -> None:
    data = np.array([[[1.0]], [[np.nan]], [[3.0]]])
    metadata: dict[str, object] = {"timestamps": np.array([10.0, 20.0, 30.0])}

    result = interpolate_nan_per_channel(data, metadata=metadata)

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 2.0, 3.0])


def test_interpolate_nan_per_channel_sorts_non_monotonic_good_timestamps() -> None:
    data = np.array([[[1.0]], [[2.0]], [[3.0]], [[np.nan]], [[5.0]]])
    metadata: dict[str, object] = {"timestamps": np.array([10.0, 20.0, 30.0, 15.0, 25.0])}

    result = interpolate_nan_per_channel(data, metadata=metadata)

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 2.0, 3.0, 1.5, 5.0])


def test_filter_quality_timesteps_can_skip_test_trimming() -> None:
    data = np.array([[[1.0]], [[0.0]], [[3.0]]])
    metadata: dict[str, object] = {"timestamps": np.array([10.0, 20.0, 30.0])}

    result = filter_quality_timesteps(
        data,
        require_positive=True,
        split="test",
        trim_test_split=False,
        metadata=metadata,
    )

    np.testing.assert_allclose(result[:, 0, 0], [1.0, 0.0, 3.0])
    np.testing.assert_array_equal(metadata["timestamps"], [10.0, 20.0, 30.0])


def test_filter_quality_timesteps_supports_min_valid_fraction() -> None:
    data = np.array(
        [
            [[10.0], [10.0], [10.0]],
            [[1.0], [10.0], [10.0]],
            [[10.0], [1.0], [1.0]],
        ]
    )
    metadata: dict[str, object] = {"timestamps": np.array([10.0, 20.0, 30.0])}

    result = filter_quality_timesteps(
        data,
        require_positive=True,
        low_quantile=10.0,
        high_quantile=100.0,
        min_valid_fraction=2.0 / 3.0,
        metadata=metadata,
    )

    np.testing.assert_allclose(result[:, :, 0], [[10.0, 10.0, 10.0], [1.0, 10.0, 10.0]])
    np.testing.assert_array_equal(metadata["timestamps"], [10.0, 20.0])


def test_keep_channel_mask_updates_metadata() -> None:
    data = np.array([[[1.0], [2.0], [3.0]]])
    metadata: dict[str, object] = {
        "channel_names": np.array(["a", "b", "c"]),
        "quality_reference_response": np.array([[[10.0], [20.0], [30.0]]]),
    }

    result = keep_channel_mask(data, keep=[True, False, True], metadata=metadata)

    np.testing.assert_allclose(result[:, :, 0], [[1.0, 3.0]])
    np.testing.assert_array_equal(metadata["channel_names"], ["a", "c"])
    np.testing.assert_allclose(metadata["quality_reference_response"][:, :, 0], [[10.0, 30.0]])


def test_keep_named_channels_uses_train_order_and_updates_metadata() -> None:
    data = np.array([[[1.0], [2.0], [3.0]]])
    metadata: dict[str, object] = {
        "channel_names": np.array(["a", "b", "c"]),
        "quality_reference_response": np.array([[[10.0], [20.0], [30.0]]]),
    }

    result = keep_named_channels(data, channel_names=["c", "a"], metadata=metadata)

    np.testing.assert_allclose(result[:, :, 0], [[3.0, 1.0]])
    np.testing.assert_array_equal(metadata["channel_names"], ["c", "a"])
    np.testing.assert_allclose(metadata["quality_reference_response"][:, :, 0], [[30.0, 10.0]])


def test_pipeline_save_requires_experiment(tmp_path: Path) -> None:
    pipeline = PreprocessingPipeline(
        {
            "output": {
                "save": True,
                "root": str(tmp_path),
                "file_name": "result.npz",
            },
            "steps": [],
        }
    )

    with pytest.raises(ValueError, match="no experiment was configured"):
        pipeline.save_output(np.zeros((2, 3, 1)), {})
