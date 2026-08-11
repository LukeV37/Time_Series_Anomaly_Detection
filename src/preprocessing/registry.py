"""Lookup tables mapping YAML names to loader and step callables."""

from __future__ import annotations

from collections.abc import Callable

from .data_loader import load_atlas_data, load_spt_data
from .transforms.filters import drop_nan_channels, drop_nan_timesteps, trim_edges
from .transforms.imputer import fill_channel_mean, fill_channel_median
from .transforms.normalizer import clip_values
from .transforms.reducer import subsample_time
from .transforms.transforms import drop_features, fill_nan, keep_features


LOADER_MAP: dict[str, Callable[..., object]] = {
    "atlas": load_atlas_data,
    "spt": load_spt_data,
}

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


def resolve_loader(loader_type: str) -> Callable[..., object]:
    try:
        return LOADER_MAP[loader_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported loader type {loader_type!r}. Available: {sorted(LOADER_MAP)}"
        ) from exc


def resolve_step(name: str) -> Callable[..., object]:
    try:
        return STEP_MAP[name]
    except KeyError as exc:
        raise ValueError(
            f"No preprocessing step {name!r}. Available: {sorted(STEP_MAP)}"
        ) from exc
