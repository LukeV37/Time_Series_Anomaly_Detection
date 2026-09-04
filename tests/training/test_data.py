from __future__ import annotations

from pathlib import Path

import numpy as np

from training.data import create_data_loaders, split_time_series, standardize_splits


def test_split_time_series_uses_chronological_raw_timesteps() -> None:
    data = np.arange(10, dtype=np.float32).reshape(10, 1, 1)

    train, val, test = split_time_series(data, train_ratio=0.6, val_ratio=0.2)

    np.testing.assert_array_equal(train[:, 0, 0], np.arange(6))
    np.testing.assert_array_equal(val[:, 0, 0], np.arange(6, 8))
    np.testing.assert_array_equal(test[:, 0, 0], np.arange(8, 10))


def test_standardize_splits_fits_only_training_timesteps() -> None:
    train = np.array([0.0, 2.0], dtype=np.float32).reshape(2, 1, 1)
    val = np.array([4.0], dtype=np.float32).reshape(1, 1, 1)
    test = np.array([6.0], dtype=np.float32).reshape(1, 1, 1)

    scaled_train, scaled_val, scaled_test = standardize_splits(train, val, test)

    np.testing.assert_allclose(scaled_train[:, 0, 0], [-1.0, 1.0])
    np.testing.assert_allclose(scaled_val[:, 0, 0], [3.0])
    np.testing.assert_allclose(scaled_test[:, 0, 0], [5.0])


def test_create_data_loaders_uses_separate_test_artifact(tmp_path: Path) -> None:
    train_path = tmp_path / "train_processed.npz"
    test_path = tmp_path / "test_processed.npz"
    np.savez_compressed(
        train_path,
        data=np.arange(12, dtype=np.float32).reshape(12, 1, 1),
        channel_names=np.array(["chan_a"]),
        years=np.array([2019]),
    )
    np.savez_compressed(
        test_path,
        data=np.arange(100, 106, dtype=np.float32).reshape(6, 1, 1),
        channel_names=np.array(["chan_a"]),
        years=np.array([2023]),
        timestamps=np.arange(6),
    )

    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        train_path,
        test_npz_path=test_path,
        window_size=2,
        batch_size=4,
        train_ratio=0.5,
        val_ratio=0.25,
        scaling="standard",
    )

    assert len(train_loader.dataset) == 5
    assert len(val_loader.dataset) == 2
    assert len(test_loader.dataset) == 5
    np.testing.assert_array_equal(metadata["test_years"], [2023])
    np.testing.assert_array_equal(metadata["test_channel_names"], ["chan_a"])
