"""Shared YAML config loading."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dictionary.

    Relative paths are first resolved from the current working directory. If that
    does not exist, paths under ``src/preprocessing`` and ``src/training`` are
    also checked so package-local config calls continue to work.
    """
    path = Path(path)
    if path.is_absolute() or path.exists():
        resolved = path
    else:
        src_root = Path(__file__).resolve().parents[1]
        candidates = [
            src_root / "preprocessing" / path,
            src_root / "training" / path,
        ]
        resolved = next((candidate for candidate in candidates if candidate.exists()), path)
    with resolved.open() as f:
        return yaml.safe_load(f)
