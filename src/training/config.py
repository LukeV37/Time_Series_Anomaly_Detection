"""Training config helpers."""

from __future__ import annotations

from pathlib import Path

from utils import load_config


def load_training_config(path: str | Path) -> dict:
    """Load a nested YAML training config."""
    return load_config(path)
