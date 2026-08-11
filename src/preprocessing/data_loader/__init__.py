"""Data loaders for preprocessing inputs."""

from .atlas import load_atlas_data
from .spt import load_spt_data

__all__ = ["load_atlas_data", "load_spt_data"]
