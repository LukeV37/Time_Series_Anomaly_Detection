#!/usr/bin/env python3
"""Run the default ATLAS preprocessing pipeline on merged CSV data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing import PreprocessingPipeline

DEFAULT_CONFIG = SRC_ROOT / "preprocessing" / "configs" / "atlas_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_number", help="ATLAS run number under $ATLAS_DATA_DIR")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline config file. Defaults to src/preprocessing/configs/atlas_pipeline.yaml.",
    )
    parser.add_argument(
        "--csv-path",
        help="Optional explicit merged CSV path. Overrides loader root/run_number resolution.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("ATLAS_DATA_DIR"),
        help="ATLAS data root. Defaults to $ATLAS_DATA_DIR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = PreprocessingPipeline.from_config_file(args.config)
    pipeline._loader_config.setdefault("params", {})["run_number"] = args.run_number
    if args.data_root:
        pipeline._loader_config["params"]["root"] = args.data_root
    if args.csv_path:
        pipeline._loader_config["params"]["csv_path"] = args.csv_path

    data, metadata = pipeline.load_and_run(run_number=args.run_number)
    print(f"processed shape: {data.shape}")
    if "output_path" in metadata:
        print(f"saved to: {metadata['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
