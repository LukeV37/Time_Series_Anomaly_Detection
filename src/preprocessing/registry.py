"""Step lookup for mapping YAML operation names to callables."""

from __future__ import annotations

from collections.abc import Callable

from .transforms.filters import drop_nan_channels, drop_nan_timesteps
from .transforms.imputer import fill_channel_median
from .transforms.normalizer import clip_values
from .transforms.transforms import fill_nan


STEP_MAP: dict[str, Callable[..., object]] = {
    "drop_nan_channels": drop_nan_channels,
    "drop_nan_timesteps": drop_nan_timesteps,
    "fill_channel_median": fill_channel_median,
    "clip_values": clip_values,
    "fill_nan": fill_nan,
}


def resolve_step(name: str) -> Callable[..., object]:
    try:
        return STEP_MAP[name]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessing step {name!r}. Available: {sorted(STEP_MAP)}"
        ) from exc
