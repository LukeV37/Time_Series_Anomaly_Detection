"""Pipeline runner for ordered NumPy preprocessing steps."""

from __future__ import annotations

import inspect

import numpy as np

from .registry import resolve_step


class PreprocessingPipeline:
    """Executes a sequence of (T, C, D) -> (T, C, D) transforms."""

    def __init__(self, config: dict) -> None:
        self._steps = []
        steps = config.get("steps", [])
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

    def run(self, data: np.ndarray, run_number: int | str | None = None) -> np.ndarray:
        result = data
        for _, fn, params, accepts_run_number in self._steps:
            if accepts_run_number:
                result = fn(result, run_number=run_number, **params)
            else:
                result = fn(result, **params)
        return result

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(steps={[label for label, _, _, _ in self._steps]})"
