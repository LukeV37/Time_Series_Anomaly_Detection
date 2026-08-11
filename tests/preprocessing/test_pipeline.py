from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.pipeline import PreprocessingPipeline


def test_pipeline_from_config_file_builds_steps() -> None:
    pipeline = PreprocessingPipeline.from_config_file("configs/spt_pipeline.yaml")

    assert repr(pipeline) == (
        "PreprocessingPipeline(steps=['filter/drop_nan_channels', "
        "'filter/drop_nan_timesteps', 'transform/fill_nan', "
        "'normalizer/clip_values'])"
    )


def test_pipeline_run_applies_steps_in_order() -> None:
    pipeline = PreprocessingPipeline(
        {
            "loader": {"type": "spt_benchmark_hdf5", "params": {}},
            "pipeline": {
                "steps": [
                    {
                        "type": "transform",
                        "function": "fill_nan",
                        "params": {"value": 1.5},
                    },
                    {
                        "type": "normalizer",
                        "function": "clip_values",
                        "params": {"low": 0.0, "high": 1.0},
                    },
                ]
            },
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
            "loader": {"type": "spt_benchmark_hdf5", "params": {}},
            "output": {
                "save": True,
                "root": str(tmp_path),
                "experiment": "preprocessing",
                "data_tag": "unit",
                "file_name": "result.npz",
            },
            "pipeline": {
                "steps": [
                    {
                        "type": "transform",
                        "function": "fill_nan",
                        "params": {"value": 0.0},
                    }
                ]
            },
        }
    )

    loaded = np.array([[[np.nan], [2.0]]])
    metadata = {
        "timestamps": [10, 20],
        "detector_names": ["a", "b"],
        "years": [2019],
        "wafer_id": "w1",
    }
    pipeline.load = lambda: (loaded, metadata)  # type: ignore[method-assign]

    result, result_metadata = pipeline.load_and_run(context={"run_number": 7})

    np.testing.assert_allclose(result, np.array([[[0.0], [2.0]]]))
    output_path = tmp_path / "preprocessing" / "unit" / "result.npz"
    assert output_path.exists()
    assert result_metadata["output_path"] == str(output_path)
    assert result_metadata["pipeline_config"] == pipeline._config["pipeline"]

    saved = np.load(output_path, allow_pickle=True)
    assert saved.files == ["data"]
    np.testing.assert_allclose(saved["data"], result)


def test_pipeline_save_requires_experiment(tmp_path: Path) -> None:
    pipeline = PreprocessingPipeline(
        {
            "output": {
                "save": True,
                "root": str(tmp_path),
                "file_name": "result.npz",
            },
            "pipeline": {"steps": []},
        }
    )

    with pytest.raises(ValueError, match="no experiment was configured"):
        pipeline._save_output(np.zeros((2, 3, 1)), {})
