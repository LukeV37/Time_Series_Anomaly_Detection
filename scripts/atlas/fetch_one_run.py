#!/usr/bin/env python3
"""Fetch one or more runs and write separate CSVs for DCM, pileup, L1A, and busy."""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from atlas.pbeast_fetcher import PBeastFetcher, parse_run_summary
from atlas.pbeast_fetcher.align import STRATEGIES

LOGGER = logging.getLogger("fetch_one_run")
SOURCES = {
    "dcm": "DCMRate",
    "pileup": "mu",
    "l1a": "L1ARate_Instant",
    "busy": "ctpcore_busy",
}


def default_html_dir() -> Path:
    html_dir = os.environ.get("PBEAST_HTML_DIR")
    if html_dir:
        return Path(html_dir)
    return REPO_ROOT / "src" / "atlas" / "pbeast_fetcher" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch one or more runs to separate CSV files.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--run-number", type=int, help="Run number to fetch.")
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Text file with one run number per line.",
    )
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT / "output",
        type=Path,
        help="Directory for CSV outputs (default: <repo>/output).",
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=default_html_dir(),
        help="Directory containing ATLASDataSummary*.html files (default: $PBEAST_HTML_DIR or bundled package data).",
    )
    parser.add_argument(
        "--merge-strategy",
        choices=sorted(STRATEGIES),
        default="s2",
        help="Alignment strategy: 's2' (fast, default) or 'baseline' (slow, for verification).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def default_config_paths() -> tuple[Path, Path]:
    return (
        REPO_ROOT / "src" / "atlas" / "pbeast_fetcher" / "configs" / "config.yaml",
        REPO_ROOT / "src" / "atlas" / "pbeast_fetcher" / "configs" / "sources.yaml",
    )


def find_run_summary(run_number: int, html_dir: Path) -> tuple[int, Path]:
    for name in sorted(html_dir.glob("ATLASDataSummary*.html"), reverse=True):
        if "_HI" in name.name:
            continue
        year = int(name.name.removeprefix("ATLASDataSummary").removesuffix(".html"))
        runs = parse_run_summary(str(name), default_year=year)
        if run_number in runs:
            return year, name
    raise SystemExit(f"Run {run_number} was not found in {html_dir}/ATLASDataSummary*.html files.")


def merged_dataframe_for_run(fetched, merge_strategy: str = "s2") -> pd.DataFrame:
    l1a_series = fetched[SOURCES["l1a"]].get_all_data()
    if not l1a_series:
        raise SystemExit(f"No data returned for source {SOURCES['l1a']}.")

    # First L1A series is the reference timeline; everything else aligns onto it.
    ref = l1a_series[0]
    source_series = (
        l1a_series[1:]
        + fetched[SOURCES["dcm"]].get_all_data()
        + fetched[SOURCES["pileup"]].get_all_data()
        + fetched[SOURCES["busy"]].get_all_data()
    )

    align = STRATEGIES[merge_strategy]
    LOGGER.info("Aligning %d source series onto reference using '%s'", len(source_series), merge_strategy)
    return align(ref, source_series)


def read_run_numbers(input_file: Path) -> list[int]:
    run_numbers = []
    for line in input_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        run_numbers.append(int(stripped))
    if not run_numbers:
        raise SystemExit(f"No run numbers found in {input_file}.")
    return run_numbers


def fetch_run(run_number: int, output_root: Path, config_path: Path, sources_path: Path, html_dir: Path, merge_strategy: str = "s2") -> None:
    output_dir = output_root / str(run_number)
    output_dir.mkdir(parents=True, exist_ok=True)

    year, html_path = find_run_summary(run_number, html_dir)
    LOGGER.info("Run %d found in %s", run_number, html_path.name)

    with PBeastFetcher.from_config(config_path, sources_path) as fetcher:
        fetched = fetcher.fetch_by_run(
            source_names=list(SOURCES.values()),
            year=year,
            run_number=run_number,
            html_path=html_path,
        )

    df = merged_dataframe_for_run(fetched, merge_strategy)
    output_path = output_dir / "merged.csv"
    df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path, sources_path = default_config_paths()
    run_numbers = [args.run_number] if args.run_number is not None else read_run_numbers(args.input_file)

    for run_number in run_numbers:
        fetch_run(run_number, args.output_dir, config_path, sources_path, args.html_dir, args.merge_strategy)


if __name__ == "__main__":
    main()
