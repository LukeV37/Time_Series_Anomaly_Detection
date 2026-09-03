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
from preprocessing.data_loader import load_spt_data
from preprocessing.transforms.filters import _stable_channel_mask, keep_channel_mask
from utils import load_config

DEFAULT_CONFIG = SRC_ROOT / "preprocessing" / "configs" / "spt_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline config file. Defaults to src/preprocessing/configs/spt_pipeline.yaml.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "test", "both"),
        default="both",
        help="Run train preprocessing, test preprocessing, or both serially.",
    )
    return parser.parse_args()


def _build_loader_params(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    loader = dict(config["loader"])
    loader_params = dict(loader.get("params", {}))
    years_key = f"{dataset}_years"
    years = loader_params.pop(years_key, None)
    if years is None:
        raise ValueError(f"SPT preprocessing config is missing loader.params.{years_key}")
    loader_params.pop("train_years", None)
    loader_params.pop("test_years", None)
    loader_params["years"] = tuple(int(year) for year in years)
    return loader_params


def _build_pipeline_config(config: dict[str, Any], *, dataset: str) -> dict[str, Any]:
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


def _run_dataset(config: dict[str, Any], dataset: str) -> tuple[Any, dict[str, Any]]:
    pipeline = PreprocessingPipeline(_build_pipeline_config(config, dataset=dataset))
    result, metadata = pipeline.load_and_run()
    return result, metadata


def _train_channel_mask(config: dict[str, Any]) -> np.ndarray | None:
    train_pipeline = PreprocessingPipeline(_build_pipeline_config(config, dataset="train"))
    train_data, train_metadata = train_pipeline.load()
    for step in train_pipeline._steps:
        params = dict(step["params"])
        if train_metadata is not None and step["supports_metadata"]:
            params.setdefault("metadata", train_metadata)
        if step["label"] == "select_stable_channels":
            reference = train_data
            reference_metadata_key = params.get("reference_metadata_key")
            if reference_metadata_key:
                reference = np.asarray(train_metadata[reference_metadata_key])
            return _stable_channel_mask(
                reference,
                low_quantile=float(params.get("low_quantile", 10.0)),
                high_quantile=float(params.get("high_quantile", 90.0)),
                tolerance=float(params.get("tolerance", 0.10)),
            )
        train_data = step["function"](train_data, **params)
    return None


def _run_test_with_train_mask(
    config: dict[str, Any], train_channel_mask: np.ndarray | None
) -> tuple[Any, dict[str, Any]]:
    pipeline_config = _build_pipeline_config(config, dataset="test")
    if train_channel_mask is not None:
        pipeline_config["steps"] = [
            step for step in pipeline_config["steps"] if step["name"] != "select_stable_channels"
        ]
    pipeline = PreprocessingPipeline(pipeline_config)
    data, metadata = pipeline.load()
    result = data
    if train_channel_mask is not None:
        result = keep_channel_mask(data, keep=train_channel_mask, metadata=metadata)
    result = pipeline.run(result, metadata=metadata)
    metadata = dict(metadata)
    metadata["pipeline_config"] = {"steps": pipeline._step_configs}
    saved_path = pipeline._save_output(result, metadata)
    if saved_path is not None:
        metadata["output_path"] = str(saved_path)
    return result, metadata


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    train_metadata: dict[str, Any] | None = None
    if args.mode in {"train", "both"}:
        train_result, train_metadata = _run_dataset(config, "train")
        print(f"train processed shape: {train_result.shape}")
        if train_metadata.get("output_path"):
            print(f"train saved to: {train_metadata['output_path']}")

    if args.mode in {"test", "both"}:
        if train_metadata is None:
            train_result, train_metadata = _run_dataset(config, "train")
            print(f"train processed shape: {train_result.shape}")
            if train_metadata.get("output_path"):
                print(f"train saved to: {train_metadata['output_path']}")
        test_result, test_metadata = _run_test_with_train_mask(
            config,
            _train_channel_mask(config),
        )
        print(f"test processed shape: {test_result.shape}")
        if test_metadata.get("output_path"):
            print(f"test saved to: {test_metadata['output_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
