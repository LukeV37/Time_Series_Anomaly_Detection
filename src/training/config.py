"""Training config helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils import load_config


def _resolve_run_dir(*, root: Any, experiment: Any, data_tag: Any) -> Path | None:
    resolved_root = root or os.environ.get("OUTPUT_DIR")
    if not resolved_root or not experiment or not data_tag:
        return None
    return Path(resolved_root) / str(experiment) / str(data_tag)


def _resolve_path(value: Any, *, base_dir: Path | None) -> Any:
    if value in {None, ""}:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if base_dir is not None:
        return str(base_dir / path)
    return str(path)


def load_training_config(path: str | Path) -> dict:
    """Load a nested YAML training config."""
    config = load_config(path)

    input_config = dict(config.get("input", {}))
    input_dir = _resolve_run_dir(
        root=input_config.get("root"),
        experiment=input_config.get("experiment"),
        data_tag=input_config.get("data_tag"),
    )
    for key in ("train_npz_path", "npz_path", "test_npz_path"):
        if key in input_config:
            input_config[key] = _resolve_path(input_config[key], base_dir=input_dir)
    config["input"] = input_config

    output_config = dict(config.get("output", {}))
    output_dir = _resolve_run_dir(
        root=output_config.get("root") or input_config.get("root"),
        experiment=input_config.get("experiment"),
        data_tag=output_config.get("data_tag") or input_config.get("data_tag"),
    )
    for key in ("checkpoint_path", "test_errors_path", "loss_curve_path"):
        if key in output_config:
            output_config[key] = _resolve_path(output_config[key], base_dir=output_dir)
    config["output"] = output_config
    return config
