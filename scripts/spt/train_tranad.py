#!/usr/bin/env python3
"""Train a minimal TranAD model on saved SPT preprocessing outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.train import train_tranad_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/training/configs/spt_tranad.yaml"),
        help="Path to nested YAML training config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _model, metrics, _checkpoint_path = train_tranad_from_config(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
