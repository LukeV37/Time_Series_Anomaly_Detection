"""YAML loading for the preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a pipeline YAML file into a plain dictionary.

    Relative paths are resolved from this package directory first so calls like
    ``load_config("configs/hlt_pipeline.yaml")`` work from anywhere.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent / path
    with path.open() as f:
        return yaml.safe_load(f)
