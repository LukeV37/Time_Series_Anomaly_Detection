#!/usr/bin/env python3
"""Plot SPT reconstruction-error histograms with high- vs low-SNR subsets.

This is a lightweight port of the histogram plotting workflow from
CrossExperimentalAIDQM. In this repository it works from saved NumPy outputs
instead of the `anldq` training/inference stack.

Expected inputs:
- an errors array with shape `(T, C)` or `(T, C, 1)`
- either explicit channel labels, or channel-wise baseline data from which
  labels can be derived using an SNR-like threshold on detector median response

Label convention matches the original script:
- label `1` means low-SNR detector
- label `0` means high-SNR detector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.data_loader.spt import load_spt_data
from training.data import load_npz_data


DEFAULT_THRESHOLD = 20.0


def return_error(all_counts: np.ndarray, subset_counts: np.ndarray) -> np.ndarray:
    """Return the uncertainty on the ratio All/Subset for a subset count."""
    all_counts = np.asarray(all_counts, dtype=float)
    subset_counts = np.asarray(subset_counts, dtype=float)
    remainder = all_counts - subset_counts
    ratio_err = np.full_like(all_counts, np.nan, dtype=float)
    valid = (subset_counts > 0) & (remainder >= 0)
    ratio_err[valid] = np.sqrt(
        remainder[valid] / subset_counts[valid] ** 2
        + remainder[valid] ** 2 / subset_counts[valid] ** 3
    )
    return ratio_err


def _ensure_time_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    raise ValueError(f"Expected shape (T, C) or (T, C, 1), got {arr.shape}")


def _load_errors(errors_path: Path) -> np.ndarray:
    errors = np.load(errors_path)
    return _ensure_time_channel(errors)


def _load_channel_labels(labels_path: Path) -> np.ndarray:
    labels = np.load(labels_path)
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels.astype(np.uint8, copy=False)
    if labels.ndim == 2:
        if labels.shape[0] == 1:
            return labels[0].astype(np.uint8, copy=False)
        first = labels[0]
        if np.all(labels == first):
            return first.astype(np.uint8, copy=False)
    raise ValueError(f"Expected labels with shape (C,) or broadcastable equivalent, got {labels.shape}")


def _derive_channel_labels_from_data(data: np.ndarray, threshold: float) -> np.ndarray:
    """Derive per-channel low-SNR labels from detector median response.

    This repo does not carry over the full CrossExperimental labeling stack, so
    we use a simple local proxy: channels with median response below `threshold`
    are labeled as low-SNR (1), otherwise high-SNR (0).
    """
    tc = _ensure_time_channel(data)
    channel_score = np.nanmedian(tc, axis=0)
    if not np.isfinite(channel_score).any():
        raise ValueError("Could not derive channel labels: all channel medians are non-finite")
    return (channel_score < float(threshold)).astype(np.uint8)


def _load_data_for_labels(args: argparse.Namespace) -> np.ndarray:
    if args.data_npz is not None:
        data, _metadata = load_npz_data(args.data_npz)
        return data
    if args.spt_root is not None or args.use_spt_loader:
        data, _metadata = load_spt_data(root=args.spt_root)
        return data
    raise ValueError(
        "Need one of --labels, --data-npz, or --use-spt-loader/--spt-root to determine channel labels."
    )


def plot_validation_error_histogram(
    errors: np.ndarray,
    channel_labels: np.ndarray,
    *,
    threshold: float,
    output_path: Path,
) -> None:
    errors = _ensure_time_channel(errors)
    channel_labels = np.asarray(channel_labels)
    if channel_labels.ndim != 1:
        raise ValueError(f"Expected channel_labels shape (C,), got {channel_labels.shape}")
    if errors.shape[1] != channel_labels.shape[0]:
        raise ValueError(
            "channel_labels length does not match errors channel count "
            f"({channel_labels.shape[0]} vs {errors.shape[1]})"
        )

    # label=1 means low-SNR; invert so blue corresponds to SNR > threshold.
    subset_mask = ~channel_labels.astype(bool)
    if subset_mask.sum() == 0 or (~subset_mask).sum() == 0:
        raise ValueError(
            "channel_labels contain only one class. Adjust the threshold or provide explicit labels."
        )

    all_vals = np.ravel(errors)
    subset_vals = np.ravel(errors[:, subset_mask])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    subset_vals = subset_vals[np.isfinite(subset_vals) & (subset_vals > 0)]
    if all_vals.size == 0 or subset_vals.size == 0:
        raise ValueError("No positive finite reconstruction errors available for plotting")

    bins = np.logspace(np.log10(min(all_vals.min(), subset_vals.min())), np.log10(all_vals.max()), 80)
    all_counts, edges = np.histogram(all_vals, bins=bins)
    subset_counts, _ = np.histogram(subset_vals, bins=bins)
    ratio = np.divide(
        all_counts,
        subset_counts,
        out=np.full_like(all_counts, np.nan, dtype=float),
        where=subset_counts > 0,
    )
    ratio_err = return_error(all_counts, subset_counts)
    centers = np.sqrt(edges[:-1] * edges[1:])
    valid = np.isfinite(ratio) & np.isfinite(ratio_err)

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    ax.hist(all_vals, bins=bins, histtype="step", label="All Detectors", color="r")
    ax.hist(
        subset_vals,
        bins=bins,
        histtype="step",
        label=f"Detectors SNR>{threshold:g}",
        color="b",
    )
    ax.set_title("Reconstruction Error Across All Timesteps")
    ax.set_ylabel("Detector Counts")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    rax.errorbar(
        centers[valid],
        ratio[valid],
        yerr=ratio_err[valid],
        fmt="o",
        color="black",
        ecolor="black",
        linestyle="none",
        markersize=3,
        elinewidth=1.5,
        capsize=2,
        capthick=1.5,
    )
    rax.axhline(1.0, color="gray", linestyle="--")
    rax.set_xscale("log")
    rax.set_xlabel("Reconstruction Error")
    rax.set_ylabel("Red/Blue")
    rax.set_ylim(0.95, 2.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True, help="Path to reconstruction errors (.npy).")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional path to explicit channel labels (.npy). Expected label 1=low-SNR, 0=high-SNR.",
    )
    parser.add_argument(
        "--data-npz",
        type=Path,
        help="Optional preprocessing output (.npz). Used to derive channel labels from detector medians.",
    )
    parser.add_argument(
        "--use-spt-loader",
        action="store_true",
        help="Load benchmark SPT data with the repo SPT loader to derive channel labels.",
    )
    parser.add_argument(
        "--spt-root",
        type=Path,
        help="Optional benchmark SPT root passed to the SPT loader when deriving labels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="SNR threshold used for labeling channels when labels are derived.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/spt_validation_error_histogram.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = _load_errors(args.errors)

    if args.labels is not None:
        channel_labels = _load_channel_labels(args.labels)
    else:
        label_data = _load_data_for_labels(args)
        channel_labels = _derive_channel_labels_from_data(label_data, threshold=args.threshold)

    plot_validation_error_histogram(
        errors,
        channel_labels,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
