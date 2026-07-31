"""Step lookup for mapping YAML step names to transform callables."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np

from .transforms.filters import drop_nan_channels, drop_nan_timesteps, trim_edges
from .transforms.imputer import fill_channel_mean, fill_channel_median
from .transforms.normalizer import apply_scale, clip_values, subtract_mean
from .transforms.reducer import subsample_time
from .transforms.transforms import drop_features, fill_nan, keep_features


TransformFn = Callable[..., np.ndarray]

STEP_MAP: dict[str, dict[str, TransformFn]] = {
    "filter": {
        "drop_nan_channels": drop_nan_channels,
        "drop_nan_timesteps": drop_nan_timesteps,
        "trim_edges": trim_edges,
    },
    "imputer": {
        "fill_channel_mean": fill_channel_mean,
        "fill_channel_median": fill_channel_median,
    },
    "normalizer": {
        "apply_scale": apply_scale,
        "clip_values": clip_values,
        "subtract_mean": subtract_mean,
    },
    "reducer": {
        "subsample_time": subsample_time,
    },
    "transform": {
        "drop_features": drop_features,
        "fill_nan": fill_nan,
        "keep_features": keep_features,
    },
}


def resolve_step(step_type: str, function_name: str) -> TransformFn:
    """Return the callable configured under ``(step_type, function_name)``."""
    try:
        return STEP_MAP[step_type][function_name]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessing step: type={step_type!r}, function={function_name!r}. "
            f"Available under {step_type!r}: {sorted(STEP_MAP.get(step_type, {}))}"
        ) from exc
