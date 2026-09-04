#!/usr/bin/env python3
"""Run the serial SPT train/test preprocessing workflow from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing import PreprocessingPipeline
from preprocessing.transforms.filters import keep_named_channels
from utils import load_config

DEFAULT_CONFIG = SRC_ROOT / "preprocessing" / "configs" / "spt_pipeline_no_trim.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline config file. Defaults to src/preprocessing/configs/spt_pipeline_no_trim.yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "test", "both"),
        default="both",
        help="Run train preprocessing, test preprocessing, or both serially.",
    )
    return parser.parse_args()


def _build_loader_params(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    loader_params = dict(config["loader"].get("params", {}))
    years_key = f"{dataset}_years"
    years = loader_params.pop(years_key, None)
    if years is None:
        raise ValueError(f"SPT preprocessing config is missing loader.params.{years_key}")
    loader_params.pop("train_years", None)
    loader_params.pop("test_years", None)
    loader_params["years"] = tuple(int(year) for year in years)
    return loader_params


def build_pipeline_config(config: dict[str, Any], *, dataset: str) -> dict[str, Any]:
    pipeline_config = {
        "loader": {"type": config["loader"]["type"], "params": _build_loader_params(config, dataset)},
        "steps": [dict(step) for step in config.get("steps", [])],
        "output": dict(config.get("output", {})),
    }
    if dataset == "test":
        pipeline_config["steps"] = [
            step for step in pipeline_config["steps"] if step["name"] != "filter_quality_timesteps"
        ]
        pipeline_config["output"]["file_name"] = "test_processed.npz"
    else:
        pipeline_config["output"]["file_name"] = "train_processed.npz"
    return pipeline_config


def _uses_train_selected_channels(config: dict[str, Any]) -> bool:
    return any(step["name"] == "select_stable_channels" for step in config.get("steps", []))


def _run_dataset(
    pipeline_config: dict[str, Any],
    *,
    data: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    selected_channel_names: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    pipeline_config = dict(pipeline_config)
    pipeline_config["steps"] = [dict(step) for step in pipeline_config.get("steps", [])]
    if selected_channel_names is not None:
        pipeline_config["steps"] = [
            step for step in pipeline_config["steps"] if step["name"] != "select_stable_channels"
        ]
    pipeline = PreprocessingPipeline(pipeline_config)
    if data is None or metadata is None:
        data, metadata = pipeline.load()
    if selected_channel_names is not None:
        data = keep_named_channels(data, channel_names=selected_channel_names, metadata=metadata)
    result = pipeline.run(data, metadata=metadata)
    metadata = dict(metadata)
    metadata["pipeline_config"] = {"steps": pipeline_config["steps"]}
    saved_path = pipeline.save_output(result, metadata)
    if saved_path is not None:
        metadata["output_path"] = str(saved_path)
    return result, metadata


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    train_metadata: dict[str, Any] | None = None
    selected_channel_names: np.ndarray | None = None
    needs_train_channels = _uses_train_selected_channels(config)

    if args.mode in {"train", "both"} or needs_train_channels:
        train_result, train_metadata = _run_dataset(build_pipeline_config(config, dataset="train"))
        if needs_train_channels:
            selected_channel_names = np.asarray(train_metadata.get("channel_names", []), dtype=str)
        if args.mode in {"train", "both"}:
            print(f"train processed shape: {train_result.shape}")
            if train_metadata.get("output_path"):
                print(f"train saved to: {train_metadata['output_path']}")

    if args.mode in {"test", "both"}:
        test_result, test_metadata = _run_dataset(
            build_pipeline_config(config, dataset="test"),
            selected_channel_names=selected_channel_names,
        )
        print(f"test processed shape: {test_result.shape}")
        if test_metadata.get("output_path"):
            print(f"test saved to: {test_metadata['output_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
