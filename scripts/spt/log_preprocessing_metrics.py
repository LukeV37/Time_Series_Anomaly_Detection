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
from utils import load_config

from run_preprocessing import _run_dataset, build_pipeline_config

DEFAULT_CONFIG = SRC_ROOT / "preprocessing" / "configs" / "spt_pipeline_no_trim.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline config file. Defaults to src/preprocessing/configs/spt_pipeline_no_trim.yaml.",
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


def _build_pipeline_config(
    config: dict[str, Any], *, dataset: str, low_quantile: float | None, high_quantile: float | None
) -> dict[str, Any]:
    pipeline_config = build_pipeline_config(config, dataset=dataset)
    if dataset == "train" and (low_quantile is not None or high_quantile is not None):
        for step in pipeline_config["steps"]:
            if step["name"] == "filter_quality_timesteps":
                params = dict(step.get("params", {}))
                if low_quantile is not None:
                    params["low_quantile"] = low_quantile
                if high_quantile is not None:
                    params["high_quantile"] = high_quantile
                step["params"] = params
                break
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


def _uses_train_selected_channels(config: dict[str, Any]) -> bool:
    return any(step["name"] == "select_stable_channels" for step in config.get("steps", []))


def _run_train(
    config: dict[str, Any], low_quantile: float | None, high_quantile: float | None
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None, dict[str, Any]]:
    pipeline_config = _build_pipeline_config(
        config,
        dataset="train",
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    pipeline = PreprocessingPipeline(pipeline_config)
    data_before, metadata_before = pipeline.load()
    before_summary = _summarize_array("train_raw", data_before, metadata_before)
    result, metadata_after = _run_dataset(pipeline_config, data=data_before, metadata=metadata_before)
    selected_channel_names = None
    if _uses_train_selected_channels(config):
        selected_channel_names = np.asarray(metadata_after.get("channel_names", []), dtype=str)
    return result, metadata_after, selected_channel_names, before_summary


def _run_test(
    config: dict[str, Any], selected_channel_names: np.ndarray | None
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    pipeline_config = _build_pipeline_config(config, dataset="test", low_quantile=None, high_quantile=None)
    pipeline = PreprocessingPipeline(pipeline_config)
    data_before, metadata_before = pipeline.load()
    before_summary = _summarize_array("test_raw", data_before, metadata_before)
    result, metadata_after = _run_dataset(
        pipeline_config,
        data=data_before,
        metadata=metadata_before,
        selected_channel_names=selected_channel_names,
    )
    return result, metadata_after, before_summary


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    train_result, train_metadata, selected_channel_names, train_before = _run_train(
        config,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )
    test_result, test_metadata, test_before = _run_test(config, selected_channel_names)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config).resolve()),
        "train_filter_quality_timesteps_override": {
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
        },
        "train_channel_mask_kept": None
        if selected_channel_names is None
        else int(selected_channel_names.shape[0]),
        "train_channel_mask_total": train_before["channels"],
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
