#!/usr/bin/env python3
"""Run SPT preprocessing and write detailed dataset metrics to a log file."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing import PreprocessingPipeline
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
        "--log-file",
        required=True,
        help="Path to the output log file.",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=None,
        help="Override filter_quality_timesteps low_quantile for the train split.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=None,
        help="Override filter_quality_timesteps high_quantile for the train split.",
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


def _build_pipeline_config(
    config: dict[str, Any], *, dataset: str, low_quantile: float | None, high_quantile: float | None
) -> dict[str, Any]:
    steps = [dict(step) for step in config.get("steps", [])]
    if dataset == "train" and (low_quantile is not None or high_quantile is not None):
        for step in steps:
            if step["name"] == "filter_quality_timesteps":
                params = dict(step.get("params", {}))
                if low_quantile is not None:
                    params["low_quantile"] = low_quantile
                if high_quantile is not None:
                    params["high_quantile"] = high_quantile
                step["params"] = params
                break

    pipeline_config = {
        "loader": {"type": config["loader"]["type"], "params": _build_loader_params(config, dataset)},
        "steps": steps,
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


def _summarize_array(name: str, data: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    timestamps = np.asarray(metadata.get("timestamps", []))
    channel_names = np.asarray(metadata.get("channel_names", []))
    sample_years = np.asarray(metadata.get("sample_years", []))
    feature_names = np.asarray(metadata.get("feature_names", []))
    summary: dict[str, Any] = {
        "name": name,
        "shape": list(data.shape),
        "timesteps": int(data.shape[0]),
        "channels": int(data.shape[1]) if data.ndim >= 2 else None,
        "features": int(data.shape[2]) if data.ndim >= 3 else None,
        "dtype": str(data.dtype),
        "finite_values": int(np.isfinite(data).sum()),
        "nan_values": int(np.isnan(data).sum()),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "median": float(np.nanmedian(data)),
        "timestamps_present": int(timestamps.shape[0]),
        "channel_names_present": int(channel_names.shape[0]),
        "feature_names": feature_names.tolist() if feature_names.size else [],
        "output_path": metadata.get("output_path"),
    }
    if timestamps.size:
        summary["first_timestamp"] = str(timestamps[0])
        summary["last_timestamp"] = str(timestamps[-1])
    if sample_years.size:
        unique_years, counts = np.unique(sample_years, return_counts=True)
        summary["sample_year_counts"] = {
            str(int(year)): int(count) for year, count in zip(unique_years, counts, strict=True)
        }
    return summary


def _compute_train_channel_mask(train_pipeline: PreprocessingPipeline) -> np.ndarray:
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
    raise ValueError("SPT train preprocessing requires a select_stable_channels step.")


def _run_train(config: dict[str, Any], low_quantile: float | None, high_quantile: float | None) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, Any]]:
    pipeline_config = _build_pipeline_config(
        config,
        dataset="train",
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    pipeline = PreprocessingPipeline(pipeline_config)
    data_before, metadata_before = pipeline.load()
    before_summary = _summarize_array("train_raw", data_before, metadata_before)
    result, metadata_after = pipeline.load_and_run()
    channel_mask = _compute_train_channel_mask(pipeline)
    return result, metadata_after, channel_mask, before_summary


def _run_test(config: dict[str, Any], train_channel_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    pipeline_config = _build_pipeline_config(config, dataset="test", low_quantile=None, high_quantile=None)
    pipeline_config["steps"] = [
        step for step in pipeline_config["steps"] if step["name"] != "select_stable_channels"
    ]
    pipeline = PreprocessingPipeline(pipeline_config)
    data_before, metadata_before = pipeline.load()
    before_summary = _summarize_array("test_raw", data_before, metadata_before)
    result = keep_channel_mask(data_before, keep=train_channel_mask, metadata=metadata_before)
    result = pipeline.run(result, metadata=metadata_before)
    metadata_after = dict(metadata_before)
    metadata_after["pipeline_config"] = {"steps": pipeline._step_configs}
    saved_path = pipeline._save_output(result, metadata_after)
    if saved_path is not None:
        metadata_after["output_path"] = str(saved_path)
    return result, metadata_after, before_summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    train_result, train_metadata, train_channel_mask, train_before = _run_train(
        config,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )
    test_result, test_metadata, test_before = _run_test(config, train_channel_mask)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "train_filter_quality_timesteps_override": {
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
        },
        "train_channel_mask_kept": int(np.count_nonzero(train_channel_mask)),
        "train_channel_mask_total": int(train_channel_mask.shape[0]),
        "datasets": {
            "train_raw": train_before,
            "train_processed": _summarize_array("train_processed", train_result, train_metadata),
            "test_raw": test_before,
            "test_processed": _summarize_array("test_processed", test_result, test_metadata),
        },
    }

    log_path = Path(args.log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    print(f"wrote preprocessing log: {log_path}")
    print(f"train processed shape: {train_result.shape}")
    print(f"test processed shape: {test_result.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
