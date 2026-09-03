#!/usr/bin/env python3
"""Plot SPT test reconstruction errors against their raw SNR values.

Defaults are resolved from the training YAML plus OUTPUT_DIR/input.experiment/output.data_tag.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.data import load_npz_data
from utils import load_config


DEFAULT_THRESHOLD = 20.0
DEFAULT_SNR_MIN = 0.0
DEFAULT_SNR_MAX = 300.0
DEFAULT_CONFIG = SRC_ROOT / "training" / "configs" / "spt_tranad.yaml"


def _resolve_output_dir(config: dict[str, Any]) -> Path:
    input_config = config.get("input", {})
    output_config = config.get("output", {})
    root = output_config.get("root") or input_config.get("root") or os.environ.get("OUTPUT_DIR")
    if not root:
        raise ValueError("Plotting requires output.root, input.root, or OUTPUT_DIR.")
    experiment = input_config.get("experiment")
    data_tag = output_config.get("data_tag") or input_config.get("data_tag")
    if not experiment or not data_tag:
        raise ValueError("Plotting config requires input.experiment and input.data_tag or output.data_tag.")
    return Path(root) / str(experiment) / str(data_tag)


def _resolve_path(value: str | Path | None, *, output_dir: Path, default_name: str) -> Path:
    if value in {None, ""}:
        return output_dir / default_name
    path = Path(value)
    if path.is_absolute():
        return path
    return output_dir / path


def _ensure_time_channel(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0]
    raise ValueError(f"Expected shape (T, C) or (T, C, 1), got {array.shape}")


def _load_test_snr(
    data_path: Path,
    *,
    error_count: int,
    window_size: int,
) -> np.ndarray:
    arrays, _metadata, used_precomputed_split = load_npz_data(data_path)
    if used_precomputed_split:
        raise ValueError("Expected standalone test preprocessing output, not precomputed split data.")

    test_snr = _ensure_time_channel(arrays["data"])
    aligned_snr = test_snr[window_size - 1 :]
    if aligned_snr.shape[0] != error_count:
        raise ValueError(
            "Test errors and raw SNR values are not aligned: "
            f"{error_count} error timesteps versus {aligned_snr.shape[0]} SNR timesteps. "
            "Use the same test preprocessing artifact and window size used during training."
        )
    return aligned_snr


def _return_error(subset_counts: np.ndarray, all_counts: np.ndarray) -> np.ndarray:
    """Return binomial uncertainty for the subset/all histogram ratio."""
    ratio_err = np.full_like(all_counts, np.nan, dtype=float)
    valid = all_counts > 0
    ratio = subset_counts[valid] / all_counts[valid]
    ratio_err[valid] = np.sqrt(ratio * (1.0 - ratio) / all_counts[valid])
    return ratio_err


def plot_joint_error_snr_histogram(
    errors: np.ndarray,
    snr: np.ndarray,
    *,
    snr_min: float,
    snr_max: float,
    output_path: Path,
) -> None:
    error_vals = np.ravel(errors)
    snr_vals = np.ravel(snr)
    valid = (
        np.isfinite(error_vals)
        & np.isfinite(snr_vals)
        & (error_vals > 0)
        & (snr_vals >= snr_min)
        & (snr_vals <= snr_max)
    )
    error_vals = error_vals[valid]
    snr_vals = snr_vals[valid]
    if error_vals.size == 0:
        raise ValueError("No positive finite reconstruction errors in the requested SNR range")

    error_bins = np.logspace(np.log10(error_vals.min()), np.log10(error_vals.max()), 151)
    snr_bins = np.linspace(snr_min, snr_max, 151)
    counts, error_edges, snr_edges = np.histogram2d(error_vals, snr_vals, bins=[error_bins, snr_bins])
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        raise ValueError("Joint histogram is empty after binning")

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(
        error_edges,
        snr_edges,
        counts.T,
        shading="auto",
        norm=LogNorm(vmin=1, vmax=positive_counts.max()),
        cmap="viridis",
    )
    ax.set_title(f"TranAD Reconstruction Error vs SNR ({snr_min:g} <= SNR <= {snr_max:g})")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("SNR")
    ax.set_xscale("log")
    ax.axhline(DEFAULT_THRESHOLD, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5)
    fig.colorbar(mesh, ax=ax, label="Detector Counts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved joint 2D plot to {output_path}")


def plot_validation_error_histogram(
    errors: np.ndarray, snr: np.ndarray, *, threshold: float, output_path: Path
) -> None:
    subset_mask = snr > threshold
    if not subset_mask.any() or not (~subset_mask).any():
        raise ValueError("SNR threshold does not split the test values into both groups")

    all_vals = np.ravel(errors)
    high_snr_vals = np.ravel(errors[subset_mask])
    low_snr_vals = np.ravel(errors[~subset_mask])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    high_snr_vals = high_snr_vals[np.isfinite(high_snr_vals) & (high_snr_vals > 0)]
    low_snr_vals = low_snr_vals[np.isfinite(low_snr_vals) & (low_snr_vals > 0)]
    if not all_vals.size or not high_snr_vals.size or not low_snr_vals.size:
        raise ValueError("No positive finite reconstruction errors available for plotting")

    bins = np.logspace(
        np.log10(min(all_vals.min(), high_snr_vals.min(), low_snr_vals.min())),
        np.log10(all_vals.max()),
        80,
    )
    all_counts, edges = np.histogram(all_vals, bins=bins)
    high_snr_counts, _ = np.histogram(high_snr_vals, bins=bins)
    ratio = np.divide(
        high_snr_counts,
        all_counts,
        out=np.full_like(high_snr_counts, np.nan, dtype=float),
        where=all_counts > 0,
    )
    ratio_err = _return_error(high_snr_counts, all_counts)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = np.isfinite(ratio) & np.isfinite(ratio_err)

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    ax.hist(all_vals, bins=bins, histtype="step", label="All Detectors", color="r")
    ax.hist(
        low_snr_vals,
        bins=bins,
        histtype="stepfilled",
        label=f"Detectors SNR<{threshold:g}",
        color="green",
        alpha=0.35,
    )
    ax.hist(high_snr_vals, bins=bins, histtype="step", label=f"Detectors SNR>{threshold:g}", color="b")
    ax.set_title("TranAD Reconstruction Error Across All Timesteps")
    ax.set_ylabel("Detector Counts")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(
        0.02,
        0.02,
        f"Fraction SNR>{threshold:g}: {subset_mask.mean():.4f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend()
    rax.errorbar(
        centers[valid], ratio[valid], yerr=ratio_err[valid], fmt="o", color="black", ecolor="black",
        linestyle="none", markersize=3, elinewidth=1.5, capsize=2, capthick=1.5,
    )
    rax.axhline(1.0, color="gray", linestyle="--")
    rax.set_xscale("log")
    rax.set_xlabel("Reconstruction Error")
    rax.set_ylabel("Blue/Red")
    rax.set_ylim(0.0, 1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training config used to resolve input/output tags and default artifact paths.",
    )
    parser.add_argument("--errors", type=Path, default=None, help="Test reconstruction errors (.npy).")
    parser.add_argument("--data-npz", type=Path, default=None, help="Raw preprocessing output (.npz).")
    parser.add_argument("--window-size", type=int, default=None, help="Training window size.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="High-SNR threshold.")
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN, help="2D plot SNR lower bound.")
    parser.add_argument("--snr-max", type=float, default=DEFAULT_SNR_MAX, help="2D plot SNR upper bound.")
    parser.add_argument("--output-1d", type=Path, default=None, help="1D histogram output PNG.")
    parser.add_argument("--output-2d", type=Path, default=None, help="2D histogram output PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = _resolve_output_dir(config)
    input_config = config.get("input", {})
    model_params = config.get("model", {}).get("params", {})

    errors_path = _resolve_path(
        args.errors or config.get("output", {}).get("test_errors_path"),
        output_dir=output_dir,
        default_name="test_errors.npy",
    )
    data_npz_path = _resolve_path(
        args.data_npz or input_config.get("test_npz_path"),
        output_dir=output_dir,
        default_name="test_processed.npz",
    )
    output_1d = _resolve_path(
        args.output_1d,
        output_dir=output_dir,
        default_name="test_error_histogram_1D.png",
    )
    output_2d = _resolve_path(
        args.output_2d,
        output_dir=output_dir,
        default_name="test_error_histogram_2D.png",
    )
    window_size = args.window_size or int(model_params.get("window_size", 10))

    errors = _ensure_time_channel(np.load(errors_path))
    snr = _load_test_snr(
        data_npz_path,
        error_count=errors.shape[0],
        window_size=window_size,
    )
    if errors.shape != snr.shape:
        raise ValueError(
            "Errors shape does not match the saved test SNR shape. "
            f"Got errors {errors.shape} versus SNR {snr.shape}. "
            "This usually means channel filtering changed the detector set during preprocessing, "
            "but the plotting input does not include the original-to-filtered channel mapping needed "
            "to align raw SNR values for the retained detectors."
        )
    plot_joint_error_snr_histogram(
        errors, snr, snr_min=args.snr_min, snr_max=args.snr_max, output_path=output_2d
    )
    plot_validation_error_histogram(errors, snr, threshold=args.threshold, output_path=output_1d)


if __name__ == "__main__":
    main()
