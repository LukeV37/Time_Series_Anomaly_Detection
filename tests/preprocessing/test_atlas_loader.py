from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.data_loader import load_atlas_csv_with_metadata


def test_load_atlas_csv_with_metadata_builds_value_and_delta_t_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "merged.csv"
    csv_path.write_text(
        "timestamp,chan_a,chan_a_deltaT,chan_b,chan_b_deltaT\n"
        "2026-01-01T00:00:00Z,1.0,0.1,10.0,1.1\n"
        "2026-01-01T00:00:01Z,2.0,0.2,20.0,1.2\n",
        encoding="ascii",
    )

    data, metadata = load_atlas_csv_with_metadata(csv_path=csv_path)

    assert data.shape == (2, 2, 2)
    np.testing.assert_allclose(data[:, 0, :], np.array([[1.0, 0.1], [2.0, 0.2]]))
    np.testing.assert_allclose(data[:, 1, :], np.array([[10.0, 1.1], [20.0, 1.2]]))
    assert metadata["source_path"] == str(csv_path.resolve())
    assert metadata["timestamps"].tolist() == ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"]
    assert metadata["channel_names"].tolist() == ["chan_a", "chan_b"]
    assert metadata["detector_names"].tolist() == ["chan_a", "chan_b"]
    assert metadata["feature_names"].tolist() == ["value", "deltaT"]
