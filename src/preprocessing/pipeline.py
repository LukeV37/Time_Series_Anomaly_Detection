"""Preprocessing pipeline for loading data and applying configured NumPy transforms."""

from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import load_atlas_data, load_spt_data
from .registry import resolve_step
from utils import load_config


LOADER_MAP = {
    "atlas": load_atlas_data,
    "spt": load_spt_data,
}


class PreprocessingPipeline:
    """Load raw data, apply configured steps, and optionally save the result."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._loader_config = config.get("loader")
        self._output_config = config.get("output", {})
        self._step_configs = list(config.get("steps", []))
        self._steps = []

        if not self._step_configs and "steps" not in config:
            raise ValueError("This pipeline config does not define top-level 'steps'.")

        for step_config in self._step_configs:
            name = step_config["name"]
            function = resolve_step(name)
            supports_metadata = "metadata" in inspect.signature(function).parameters
            self._steps.append(
                {
                    "label": name,
                    "function": function,
                    "params": dict(step_config.get("params", {})),
                    "supports_metadata": supports_metadata,
                }
            )

    @classmethod
    def from_config_file(cls, path: str | Path) -> "PreprocessingPipeline":
        """Build a pipeline from a YAML config file."""
        return cls(load_config(path))

    def load(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Load input data using the configured loader."""
        if self._loader_config is None:
            raise ValueError("This pipeline config does not define a loader.")

        loader_type = self._loader_config["type"]
        loader_params = dict(self._loader_config.get("params", {}))
        loader = LOADER_MAP.get(loader_type)
        if loader is None:
            raise ValueError(
                f"Unsupported loader type {loader_type!r}. "
                f"Expected one of {sorted(LOADER_MAP)}."
            )

        if loader_type == "spt" and "years" in loader_params:
            loader_params["years"] = tuple(int(year) for year in loader_params["years"])

        return loader(**loader_params)

    def run(self, data: np.ndarray, metadata: dict[str, Any] | None = None) -> np.ndarray:
        """Apply each configured step in order to an already-loaded array."""
        result = data
        for step in self._steps:
            params = dict(step["params"])
            if metadata is not None and step["supports_metadata"]:
                params.setdefault("metadata", metadata)
            result = step["function"](result, **params)
        return result

    def load_and_run(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Load input data, apply the configured steps, and optionally save output."""
        data, metadata = self.load()
        result = self.run(data, metadata=metadata)
        metadata = dict(metadata)
        metadata["pipeline_config"] = {"steps": self._step_configs}
        saved_path = self._save_output(result, metadata)
        if saved_path is not None:
            metadata["output_path"] = str(saved_path)
        return result, metadata

    def _save_output(self, data: np.ndarray, metadata: dict[str, Any]) -> Path | None:
        if not self._output_config.get("save", False):
            return None

        if data.ndim != 3:
            raise ValueError(f"Expected final preprocessing output shape (T, C, F), got {data.shape}")

        root = self._output_config.get("root") or os.environ.get("OUTPUT_DIR")
        if not root:
            raise ValueError("Output saving requested but no output root was configured.")

        experiment = self._output_config.get("experiment")
        if not experiment:
            raise ValueError("Output saving requested but no experiment was configured.")

        data_tag = self._output_config.get("data_tag", "default")
        file_name = self._output_config.get("file_name", "processed.npz")
        output_dir = Path(root) / experiment / data_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_name

        np.savez_compressed(output_path, data=data)
        return output_path

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(steps={[step['label'] for step in self._steps]})"
