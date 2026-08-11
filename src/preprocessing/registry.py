"""Step lookup for mapping YAML operation names to callables."""

from __future__ import annotations

from collections.abc import Callable

from .transforms.filters import drop_nan_channels, drop_nan_timesteps, trim_edges
from .transforms.imputer import fill_channel_mean, fill_channel_median
from .transforms.normalizer import clip_values
from .transforms.reducer import subsample_time
from .transforms.transforms import drop_features, fill_nan, keep_features


STEP_MAP: dict[str, Callable[..., object]] = {
    "drop_nan_channels": drop_nan_channels,
    "drop_nan_timesteps": drop_nan_timesteps,
    "trim_edges": trim_edges,
    "fill_channel_median": fill_channel_median,
    "fill_channel_mean": fill_channel_mean,
    "clip_values": clip_values,
    "fill_nan": fill_nan,
    "subsample_time": subsample_time,
    "drop_features": drop_features,
    "keep_features": keep_features,
}


def resolve_step(name: str) -> Callable[..., object]:
    try:
        return STEP_MAP[name]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessing step {name!r}. Available: {sorted(STEP_MAP)}"
        ) from exc
