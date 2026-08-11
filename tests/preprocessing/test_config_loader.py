from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config


def test_load_config_resolves_preprocessing_config_from_src() -> None:
    config = load_config("configs/spt_pipeline.yaml")

    assert config["loader"]["type"] == "spt"
    assert config["steps"][0]["name"] == "drop_nan_channels"


def test_load_config_resolves_atlas_preprocessing_config_from_src() -> None:
    config = load_config("configs/atlas_pipeline.yaml")

    assert config["loader"]["type"] == "atlas"
    assert config["steps"][0]["name"] == "drop_nan_channels"


def test_load_config_reads_absolute_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("value: 123\nname: demo\n", encoding="ascii")

    config = load_config(config_path)

    assert config == {"value": 123, "name": "demo"}
