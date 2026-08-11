from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.pipeline import PreprocessingPipeline


def test_atlas_pipeline_from_default_config_drops_sparse_channel_and_imputes(tmp_path: Path) -> None:
    csv_path = tmp_path / "merged.csv"
    csv_path.write_text(
        "timestamp,chan_a,chan_a_deltaT,chan_b,chan_b_deltaT\n"
        "2026-01-01T00:00:00Z,1.0,0.1,10.0,1.0\n"
        "2026-01-01T00:00:01Z,2.0,0.2,,\n"
        "2026-01-01T00:00:02Z,3.0,0.3,,\n"
        "2026-01-01T00:00:03Z,4.0,0.4,,\n",
        encoding="ascii",
    )

    pipeline = PreprocessingPipeline.from_config_file("configs/atlas_pipeline.yaml")
    pipeline._loader_config["params"]["csv_path"] = str(csv_path)

    result, metadata = pipeline.load_and_run()

    assert result.shape == (4, 1, 2)
    np.testing.assert_allclose(result[:, 0, :], np.array([[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]]))
    assert not np.isnan(result).any()
    assert metadata["feature_names"].tolist() == ["value", "deltaT"]
    assert metadata["channel_names"].tolist() == ["chan_a", "chan_b"]
