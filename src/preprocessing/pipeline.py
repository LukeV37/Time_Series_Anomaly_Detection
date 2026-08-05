"""Pipeline runner for loader-driven NumPy preprocessing steps."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import load_spt_benchmark_hdf5_with_metadata
from .registry import resolve_step
from utils import load_config


class PreprocessingPipeline:
    """Loads data, then executes a sequence of (T, C, D) -> (T, C, D) transforms."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._loader_config = config["loader"]
        self._output_config = config.get("output", {})
        self._steps = []
        steps = config.get("pipeline", {}).get("steps", [])
        for step in steps:
            fn = resolve_step(step["type"], step["function"])
            self._steps.append(
                (
                    f"{step['type']}/{step['function']}",
                    fn,
                    step.get("params", {}),
                    "run_number" in inspect.signature(fn).parameters,
                )
            )

    @classmethod
    def from_config_file(cls, path: str | Path) -> "PreprocessingPipeline":
        """Build a pipeline from a YAML config file."""
        return cls(load_config(path))

    def load(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Load input data using the configured loader."""
        loader_type = self._loader_config["type"]
        loader_params = dict(self._loader_config.get("params", {}))
        if loader_type != "spt_benchmark_hdf5":
            raise ValueError(
                f"Unsupported loader type {loader_type!r}. "
                "Expected 'spt_benchmark_hdf5'."
            )

        years = loader_params.get("years")
        if years is not None:
            loader_params["years"] = tuple(int(year) for year in years)

        return load_spt_benchmark_hdf5_with_metadata(**loader_params)

    def run(self, data: np.ndarray, run_number: int | str | None = None) -> np.ndarray:
        result = data
        for _, fn, params, accepts_run_number in self._steps:
            if accepts_run_number:
                result = fn(result, run_number=run_number, **params)
            else:
                result = fn(result, **params)
        return result

    def load_and_run(
        self,
        *,
        run_number: int | str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Load input data using the configured loader and run pipeline steps."""
        data, metadata = self.load()
        result = self.run(data, run_number=run_number)
        metadata = dict(metadata)
        metadata["pipeline_config"] = self._config["pipeline"]
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
        return f"PreprocessingPipeline(steps={[label for label, _, _, _ in self._steps]})"
