"""Preprocessing pipeline for loading data and applying configured NumPy transforms."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import load_atlas_csv_with_metadata, load_spt_benchmark_hdf5_with_metadata
from .registry import resolve_step
from utils import load_config


LOADER_MAP = {
    "atlas_csv": load_atlas_csv_with_metadata,
    "spt_benchmark_hdf5": load_spt_benchmark_hdf5_with_metadata,
}


class PreprocessingPipeline:
    """Load raw data, apply configured steps, and optionally save the result."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._loader_config = config.get("loader")
        self._output_config = config.get("output", {})
        self._pipeline_config = config.get("pipeline", {})
        self._steps = []

        for step_config in self._pipeline_config.get("steps", []):
            step_type = step_config["type"]
            function_name = step_config["function"]
            self._steps.append(
                {
                    "label": f"{step_type}/{function_name}",
                    "function": resolve_step(step_type, function_name),
                    "params": dict(step_config.get("params", {})),
                    "uses_context": bool(step_config.get("uses_context", False)),
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

        if loader_type == "spt_benchmark_hdf5" and "years" in loader_params:
            loader_params["years"] = tuple(int(year) for year in loader_params["years"])

        return loader(**loader_params)

    def run(self, data: np.ndarray, context: dict[str, Any] | None = None) -> np.ndarray:
        """Apply each configured step in order to an already-loaded array."""
        result = data
        step_context = dict(context or {})
        for step in self._steps:
            params = dict(step["params"])
            if step["uses_context"]:
                params["context"] = step_context
            result = step["function"](result, **params)
        return result

    def load_and_run(
        self,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Load input data, apply the configured steps, and optionally save output."""
        data, metadata = self.load()
        merged_context = dict(metadata)
        if context:
            merged_context.update(context)
        result = self.run(data, context=merged_context)
        metadata = dict(metadata)
        metadata["pipeline_config"] = self._pipeline_config
        saved_path = self._save_output(result, metadata)
        if saved_path is not None:
            metadata["output_path"] = str(saved_path)
        return result, metadata

    def _save_output(self, data: np.ndarray, metadata: dict[str, Any]) -> Path | None:
        if not self._output_config.get("save", False):
            return None

        root = self._output_config.get("root") or os.environ.get("OUTPUT_DIR")
        if not root:
            raise ValueError("Output saving requested but no output root was configured.")

        experiment = self._output_config.get("experiment", "spt")
        data_tag = self._output_config.get("data_tag", "default")
        file_name = self._output_config.get("file_name", "processed.npz")
        output_dir = Path(root) / experiment / data_tag
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_name

        np.savez_compressed(
            output_path,
            data=data,
            timestamps=np.asarray(metadata.get("timestamps", [])),
            detector_names=np.asarray(metadata.get("detector_names", []), dtype=object),
            years=np.asarray(metadata.get("years", [])),
            wafer_id=np.asarray(metadata.get("wafer_id", "")),
        )
        return output_path

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(steps={[step['label'] for step in self._steps]})"
