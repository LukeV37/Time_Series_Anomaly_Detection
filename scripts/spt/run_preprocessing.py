#!/usr/bin/env python3
"""Run the SPT preprocessing pipeline from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing import PreprocessingPipeline

DEFAULT_CONFIG = SRC_ROOT / "preprocessing" / "configs" / "spt_pipeline.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline config file. Defaults to src/preprocessing/configs/spt_pipeline.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = PreprocessingPipeline.from_config_file(args.config)
    result, metadata = pipeline.load_and_run()

    if isinstance(result, dict):
        for split_name in ("train", "val", "test"):
            split_payload = result.get(split_name)
            if split_payload is not None:
                print(f"{split_name} shape: {split_payload['data'].shape}")
    else:
        print(f"processed shape: {result.shape}")

    output_path = metadata.get("artifacts", {}).get("output_path")
    if output_path:
        print(f"saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
