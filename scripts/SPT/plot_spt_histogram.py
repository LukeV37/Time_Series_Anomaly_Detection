import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import anldq.deep_learning  # registers models in the registry
from anldq.configs import TrainConfig
from anldq.datasets import load_dataset
from anldq.trainer import build_trainer
from anldq.infer import build_infer

# SPT-specific example settings.
DATASET = "spt"
LABEL_SNR_THRESHOLD = 20.0

# Set to False to skip training + inference and jump straight to plotting.
# The run directory is derived from the config as usual, so the prior results
# must already exist on disk.
RUN_TRAINING = True


def _get_phase2_checkpoint_epochs(cfg: TrainConfig) -> list[int]:
    """Return saved checkpoint epochs that fall in adversarial phase-2."""
    adv_cfg = getattr(cfg, "tranad_adv", None)
    adv_start_epoch = getattr(adv_cfg, "adv_start_epoch", None)
    if adv_start_epoch is None:
        return []

    ckpt_dir = cfg.io.get_checkpoint_dir(cfg.run_id)
    epochs: list[int] = []
    for ckpt_path in sorted(ckpt_dir.glob("model_[0-9][0-9][0-9][0-9].ckpt")):
        stem = ckpt_path.stem
        try:
            epoch = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if epoch >= adv_start_epoch:
            epochs.append(epoch)
    return epochs


def _save_phase2_checkpoint_errors(cfg: TrainConfig, test_ds, run_dir: Path) -> Path:
    """Run inference for each saved phase-2 checkpoint and save errors locally."""
    output_dir = run_dir / "phase2_checkpoint_errors"
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = _get_phase2_checkpoint_epochs(cfg)
    if not epochs:
        print("[Phase2] No saved phase-2 checkpoints found for per-epoch inference.")
        return output_dir

    for epoch in epochs:
        infer = build_infer(cfg, load_best=False, load_epoch=epoch)
        infer.run(test_ds, save_dir=output_dir)
        errors = infer._last_errors
        if errors is None:
            print(f"[Phase2] No errors returned for epoch {epoch:04d}; skipping save.")
            continue
        errors_path = output_dir / f"errors_phase2_epoch_{epoch:04d}.npy"
        np.save(errors_path, errors)
        print(f"[Phase2] saved {errors_path.name} ({errors.shape}) → {errors_path}")

    return output_dir


def return_error(all_counts, subset_counts):
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


def plot_validation_error_histogram_v2(
    run_dir: Path,
    channel_labels: np.ndarray,
    *,
    threshold: float,
    output_path: Path,
    errors_path: Path | None = None,
) -> None:
    """Like plot_validation_error_histogram but with two additions:

    1. A green step-filled histogram showing the *difference* (red minus blue),
       i.e. the counts that are in "All Detectors" but not in the high-SNR
       subset — effectively the SNR < threshold population.
    2. The ratio panel y-axis is zoomed to [0.95, 1.02] to highlight fine
       structure near unity.
    """
    if errors_path is None:
        errors_path = run_dir / "errors.npy"
    if not errors_path.exists():
        print(f"Skipping validation plot v2: missing errors file at {errors_path}")
        return

    errors = np.load(errors_path)
    channel_labels = np.asarray(channel_labels)
    if errors.shape != channel_labels.shape:
        print(
            "Skipping validation plot v2: channel_labels shape does not match errors "
            f"({channel_labels.shape} vs {errors.shape})."
        )
        return

    # channel_labels use 1 for low-SNR bins; invert locally so the plotted blue
    # subset matches the historical legend text (channels with SNR > threshold).
    subset_mask = ~channel_labels.astype(bool)
    if subset_mask.sum() == 0 or (~subset_mask).sum() == 0:
        print(
            "Skipping validation plot v2: channel_labels contain only one class "
            f"{np.unique(channel_labels.astype(int)).tolist()}. Adjust the SPT SNR labeling "
            "threshold to produce both high- and low-SNR channel bins."
        )
        return

    all_vals = np.ravel(errors)
    subset_vals = np.ravel(errors[subset_mask])
    # low-SNR (anomaly) samples — the complement of subset_mask
    low_snr_vals = np.ravel(errors[~subset_mask])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    subset_vals = subset_vals[np.isfinite(subset_vals) & (subset_vals > 0)]
    low_snr_vals = low_snr_vals[np.isfinite(low_snr_vals) & (low_snr_vals > 0)]
    if all_vals.size == 0 or subset_vals.size == 0:
        print("Skipping validation plot v2: no positive finite reconstruction errors available.")
        return

    bins = np.logspace(np.log10(min(all_vals.min(), subset_vals.min())), np.log10(all_vals.max()), 80)
    all_counts, edges = np.histogram(all_vals, bins=bins)
    subset_counts, _ = np.histogram(subset_vals, bins=bins)
    low_snr_counts, _ = np.histogram(low_snr_vals, bins=bins)
    ratio = np.divide(
        subset_counts,
        all_counts,
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
    # Green difference histogram: counts from the SNR < threshold population,
    # drawn as a step-filled area so it is visually distinct.
    ax.stairs(
        low_snr_counts,
        edges,
        fill=True,
        color="green",
        alpha=0.35,
        label=f"Detectors SNR<{threshold:g} (difference)",
        zorder=1,
    )
    ax.stairs(
        low_snr_counts,
        edges,
        fill=False,
        color="green",
        linewidth=1.0,
        zorder=2,
    )
    ax.set_title("TranAD Reconstruction Error Across All Timesteps")
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
    rax.set_ylabel("Blue/Red")
    rax.set_ylim(0.97, 1.01)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation plot v2 to {output_path}")


def plot_validation_error_histogram(
    run_dir: Path,
    channel_labels: np.ndarray,
    *,
    threshold: float,
    output_path: Path,
    errors_path: Path | None = None,
) -> None:
    if errors_path is None:
        errors_path = run_dir / "errors.npy"
    if not errors_path.exists():
        print(f"Skipping validation plot: missing errors file at {errors_path}")
        return

    errors = np.load(errors_path)
    channel_labels = np.asarray(channel_labels)
    if errors.shape != channel_labels.shape:
        print(
            "Skipping validation plot: channel_labels shape does not match errors "
            f"({channel_labels.shape} vs {errors.shape})."
        )
        return

    # channel_labels use 1 for low-SNR bins; invert locally so the plotted blue
    # subset matches the historical legend text (channels with SNR > threshold).
    subset_mask = ~channel_labels.astype(bool)
    if subset_mask.sum() == 0 or (~subset_mask).sum() == 0:
        print(
            "Skipping validation plot: channel_labels contain only one class "
            f"{np.unique(channel_labels.astype(int)).tolist()}. Adjust the SPT SNR labeling "
            "threshold to produce both high- and low-SNR channel bins."
        )
        return

    all_vals = np.ravel(errors)
    subset_vals = np.ravel(errors[subset_mask])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    subset_vals = subset_vals[np.isfinite(subset_vals) & (subset_vals > 0)]
    if all_vals.size == 0 or subset_vals.size == 0:
        print("Skipping validation plot: no positive finite reconstruction errors available.")
        return

    bins = np.logspace(np.log10(min(all_vals.min(), subset_vals.min())), np.log10(all_vals.max()), 80)
    all_counts, edges = np.histogram(all_vals, bins=bins)
    subset_counts, _ = np.histogram(subset_vals, bins=bins)
    ratio = np.divide(
        subset_counts,
        all_counts,
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
    ax.set_title("TranAD Reconstruction Error Across All Timesteps")
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
    rax.set_ylabel("Blue/Red")
    rax.set_ylim(0.0, 1.1)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved validation plot to {output_path}")


def compute_roc_scan(
    run_dir: Path,
    labels: np.ndarray,
    *,
    threshold_min: float = 1e-3,
    n_points: int = 500,
    output_prefix: str = "roc_scan",
    errors_path: Path | None = None,
) -> None:
    """Scan reconstruction-error thresholds and write TPR/FPR arrays to disk.

    For each threshold t (log-spaced from *threshold_min* to the maximum
    finite reconstruction error):
      - Predicted positive: error >= t
      - TPR = TP / (TP + FN)  [label == 1 rows]
      - FPR = FP / (FP + TN)  [label == 0 rows]

    Outputs (written to *run_dir*):
      <output_prefix>.npz  — arrays: thresholds, tpr, fpr
      <output_prefix>.csv  — three-column CSV: threshold, tpr, fpr

    Parameters
    ----------
    run_dir:
        Directory where outputs are written.
    labels:
        Per-element label array matching the shape of the errors file.
        1 = anomaly (positive class), 0 = normal (negative class).
    threshold_min:
        Lower bound of the log-spaced threshold scan (default 1e-3).
    n_points:
        Number of threshold values to evaluate (default 500).
    output_prefix:
        Stem name for output files (default ``"roc_scan"``).
    errors_path:
        Explicit path to the errors ``.npy`` file.  When *None* (default)
        the function looks for ``errors.npy`` inside *run_dir*.
    """
    if errors_path is None:
        errors_path = run_dir / "errors.npy"
    if not errors_path.exists():
        print(f"compute_roc_scan: missing errors file at {errors_path}")
        return

    errors = np.load(errors_path)
    labels = np.asarray(labels)

    if errors.shape != labels.shape:
        print(
            f"compute_roc_scan: labels shape {labels.shape} does not match "
            f"errors shape {errors.shape} — aborting."
        )
        return

    errors_flat = errors.ravel().astype(float)
    labels_flat = labels.ravel().astype(int)

    # Keep only finite, positive errors; propagate label alignment.
    finite_mask = np.isfinite(errors_flat) & (errors_flat > 0)
    errors_flat = errors_flat[finite_mask]
    labels_flat = labels_flat[finite_mask]

    if errors_flat.size == 0:
        print("compute_roc_scan: no positive finite reconstruction errors — aborting.")
        return

    pos_mask = labels_flat == 1   # anomaly
    neg_mask = labels_flat == 0   # normal
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        print(
            f"compute_roc_scan: need both classes present (n_pos={n_pos}, n_neg={n_neg}) — aborting."
        )
        return

    threshold_max = errors_flat.max()
    if threshold_min >= threshold_max:
        print(
            f"compute_roc_scan: threshold_min ({threshold_min:.3g}) >= "
            f"max error ({threshold_max:.3g}) — aborting."
        )
        return

    thresholds = np.logspace(
        np.log10(threshold_min),
        np.log10(threshold_max),
        n_points,
    )

    tpr = np.empty(n_points, dtype=float)
    fpr = np.empty(n_points, dtype=float)

    for i, t in enumerate(thresholds):
        predicted_pos = errors_flat >= t
        tpr[i] = predicted_pos[pos_mask].sum() / n_pos
        fpr[i] = predicted_pos[neg_mask].sum() / n_neg

    # Write numpy archive
    npz_path = run_dir / f"{output_prefix}.npz"
    np.savez(npz_path, thresholds=thresholds, tpr=tpr, fpr=fpr)
    print(f"compute_roc_scan: saved arrays to {npz_path}")

    # Write CSV
    csv_path = run_dir / f"{output_prefix}.csv"
    header = "threshold,tpr,fpr"
    np.savetxt(
        csv_path,
        np.column_stack([thresholds, tpr, fpr]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8g",
    )
    print(f"compute_roc_scan: saved CSV to {csv_path}")


def plot_roc_curve(
    run_dir: Path,
    *,
    roc_prefix: str = "roc_scan",
    output_path: Path | None = None,
    cmap: str = "plasma",
    title: str = "ROC Curve — TranAD Reconstruction Error Threshold Scan",
    npz_path: Path | None = None,
) -> None:
    """Plot a TPR-vs-FPR ROC curve with threshold encoded as a color scale.

    Reads the ``<roc_prefix>.npz`` file produced by :func:`compute_roc_scan`
    from *run_dir*.  Each point on the curve is coloured by its corresponding
    (log-scale) threshold value so all three dimensions are visible at once.

    Parameters
    ----------
    run_dir:
        Directory used to resolve the default npz path and output path.
    roc_prefix:
        Stem name used when the scan was saved (default ``"roc_scan"``).
    output_path:
        Where to write the PNG.  Defaults to
        ``run_dir / "<roc_prefix>_plot.png"``.
    cmap:
        Matplotlib colormap for the threshold colour scale (default
        ``"plasma"``).  Any perceptually-uniform map works well; ``"viridis"``
        and ``"inferno"`` are good alternatives.
    title:
        Axes title string.
    npz_path:
        Explicit path to the ``.npz`` scan file.  When *None* (default) the
        function looks for ``<roc_prefix>.npz`` inside *run_dir*.
    """
    if npz_path is None:
        npz_path = run_dir / f"{roc_prefix}.npz"
    if not npz_path.exists():
        print(f"plot_roc_curve: missing scan file at {npz_path}")
        return

    data = np.load(npz_path)
    thresholds = data["thresholds"]
    tpr = data["tpr"]
    fpr = data["fpr"]

    if output_path is None:
        output_path = run_dir / f"{roc_prefix}_plot.png"

    # Color by log10(threshold) so the color scale is perceptually linear.
    log_thresh = np.log10(thresholds)
    norm = plt.Normalize(vmin=log_thresh.min(), vmax=log_thresh.max())
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw the curve as a collection of coloured line segments so each
    # segment inherits the colour of its lower-threshold endpoint.
    points = np.column_stack([fpr, tpr])           # (N, 2)
    segments = np.stack([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)

    from matplotlib.collections import LineCollection

    lc = LineCollection(
        segments,
        cmap=cmap_obj,
        norm=norm,
        linewidth=2,
        zorder=2,
    )
    lc.set_array(log_thresh[:-1])   # colour each segment by its start threshold
    ax.add_collection(lc)

    # Scatter the individual points (smaller, same colour) for hover clarity.
    sc = ax.scatter(
        fpr,
        tpr,
        c=log_thresh,
        cmap=cmap_obj,
        norm=norm,
        s=8,
        zorder=3,
        linewidths=0,
    )

    # Reference diagonal (random classifier).
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, zorder=1, label="Random")

    # Colorbar — label shows actual threshold values at ticks.
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Reconstruction error threshold (log₁₀ scale)", fontsize=10)

    # Add a second set of tick labels showing the real threshold values.
    log_ticks = cbar.get_ticks()
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{t:.1f}}}$" for t in log_ticks])

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"plot_roc_curve: saved plot to {output_path}")


def plot_phase_roc_comparison(
    run_dir: Path,
    *,
    phase1_prefix: str = "roc_scan_phase1",
    phase2_prefix: str = "roc_scan_phase2",
    output_path: Path | None = None,
    figsize: tuple[float, float] = (7, 6),
    dpi: int = 200,
) -> None:
    """Overlay phase-1 (blue) and phase-2 (red) ROC curves on a single axes.

    Phase-1 is drawn in solid blue, phase-2 in solid red.  No colorbar is
    shown — the two phases are distinguished purely by colour.  AUC values
    appear in the legend.  The figure is saved to *output_path* (default:
    ``run_dir / "roc_phase_comparison.png"``).

    Parameters
    ----------
    run_dir:
        Directory containing the ``<phase1_prefix>.npz`` and
        ``<phase2_prefix>.npz`` files produced by :func:`compute_roc_scan`.
    phase1_prefix:
        NPZ stem for phase-1 scan (default ``"roc_scan_phase1"``).
    phase2_prefix:
        NPZ stem for phase-2 scan (default ``"roc_scan_phase2"``).
    output_path:
        Where to write the PNG.  Defaults to
        ``run_dir / "roc_phase_comparison.png"``.
    figsize:
        Figure size in inches ``(width, height)``.
    dpi:
        Output resolution (default 200).
    """
    from matplotlib.lines import Line2D

    if output_path is None:
        output_path = run_dir / "roc_phase_comparison.png"

    phases = [
        (phase1_prefix, "Phase 1 (cooperative pre-training)", "steelblue"),
        (phase2_prefix, "Phase 2 (adversarial)",              "crimson"),
    ]

    # Load both files; skip gracefully if either is absent.
    loaded: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for prefix, label, color in phases:
        npz_path = run_dir / f"{prefix}.npz"
        if not npz_path.exists():
            print(f"plot_phase_roc_comparison: missing {npz_path} — skipping phase.")
            continue
        data = np.load(npz_path)
        missing = [k for k in ("thresholds", "tpr", "fpr") if k not in data]
        if missing:
            print(f"plot_phase_roc_comparison: {npz_path} missing keys {missing} — skipping.")
            continue
        loaded.append((prefix, label, color, data["thresholds"], data["tpr"], data["fpr"]))

    if not loaded:
        print("plot_phase_roc_comparison: no usable phase scan files found — aborting.")
        return

    if len(loaded) == 1:
        print(
            "plot_phase_roc_comparison: only one phase file found — "
            "drawing single-curve comparison anyway."
        )

    fig, ax = plt.subplots(figsize=figsize)

    # Random-classifier diagonal
    ax.plot(
        [-0.02, 1], [-0.02, 1],
        color="0.75", linestyle="--", linewidth=1, zorder=1, label="Random",
    )

    for _prefix, label, color, thresholds, tpr, fpr in loaded:
        # AUC via trapezoidal rule (sort by FPR for correctness).
        order = np.argsort(fpr)
        auc = float(np.trapezoid(tpr[order], fpr[order]))

        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2.2,
            zorder=2,
            label=f"{label}  (AUC = {auc:.3f})",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("Phase 1 vs Phase 2 ROC Curve Comparison", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"plot_phase_roc_comparison: saved plot to {output_path}")


def plot_phase_roc_comparison_zoomed(
    run_dir: Path,
    *,
    phase1_prefix: str = "roc_scan_phase1",
    phase2_prefix: str = "roc_scan_phase2",
    output_path: Path | None = None,
    fpr_max: float = 0.05,
    figsize: tuple[float, float] = (7, 6),
    dpi: int = 200,
) -> None:
    """Overlay phase-1 (blue) and phase-2 (red) ROC curves zoomed to low FPR.

    Identical to :func:`plot_phase_roc_comparison` except the x-axis is
    restricted to ``[0, fpr_max]`` (default 0.05) to highlight behaviour at
    the highest reconstruction-error thresholds, where the classifier is most
    selective.

    AUC in the legend is partial AUC over ``[0, fpr_max]`` (not normalised).
    The figure is saved to *output_path* (default:
    ``run_dir / "roc_phase_comparison_zoom.png"``).

    Parameters
    ----------
    run_dir:
        Directory containing the ``<phase1_prefix>.npz`` and
        ``<phase2_prefix>.npz`` files produced by :func:`compute_roc_scan`.
    phase1_prefix:
        NPZ stem for phase-1 scan (default ``"roc_scan_phase1"``).
    phase2_prefix:
        NPZ stem for phase-2 scan (default ``"roc_scan_phase2"``).
    output_path:
        Where to write the PNG.  Defaults to
        ``run_dir / "roc_phase_comparison_zoom.png"``.
    fpr_max:
        Upper bound on FPR for the zoomed view (default 0.05).
    figsize:
        Figure size in inches ``(width, height)``.
    dpi:
        Output resolution (default 200).
    """
    if output_path is None:
        output_path = run_dir / "roc_phase_comparison_zoom.png"

    phases = [
        (phase1_prefix, "Phase 1 (cooperative pre-training)", "steelblue"),
        (phase2_prefix, "Phase 2 (adversarial)",              "crimson"),
    ]

    # Load both files; skip gracefully if either is absent.
    loaded: list[tuple[str, str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for prefix, label, color in phases:
        npz_path = run_dir / f"{prefix}.npz"
        if not npz_path.exists():
            print(f"plot_phase_roc_comparison_zoomed: missing {npz_path} — skipping phase.")
            continue
        data = np.load(npz_path)
        missing = [k for k in ("thresholds", "tpr", "fpr") if k not in data]
        if missing:
            print(f"plot_phase_roc_comparison_zoomed: {npz_path} missing keys {missing} — skipping.")
            continue
        loaded.append((prefix, label, color, data["thresholds"], data["tpr"], data["fpr"]))

    if not loaded:
        print("plot_phase_roc_comparison_zoomed: no usable phase scan files found — aborting.")
        return

    if len(loaded) == 1:
        print(
            "plot_phase_roc_comparison_zoomed: only one phase file found — "
            "drawing single-curve comparison anyway."
        )

    # Pre-compute masked data for all phases so we can derive the TPR range
    # before drawing anything.
    #
    # pAUC is computed on a shared FPR grid anchored at [0, fpr_max] so that
    # both curves are integrated over exactly the same x-interval.  Without
    # this, the raw threshold scan may land at slightly different FPR values
    # near fpr_max, causing the final trapezoid to have different widths and
    # producing a spurious flip in the pAUC ranking.
    _N_GRID = 10_000
    _fpr_grid = np.linspace(0.0, fpr_max, _N_GRID)

    masked: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []
    for _prefix, label, color, thresholds, tpr, fpr in loaded:
        mask = fpr <= fpr_max
        fpr_z = fpr[mask]
        tpr_z = tpr[mask]
        order = np.argsort(fpr_z)
        fpr_sorted = fpr_z[order]
        tpr_sorted = tpr_z[order]
        # Interpolate onto the shared grid; np.interp clamps to boundary values.
        tpr_interp = np.interp(_fpr_grid, fpr_sorted, tpr_sorted)
        partial_auc = float(np.trapezoid(tpr_interp, _fpr_grid))
        masked.append((label, color, fpr_z, tpr_z, partial_auc))

    # TPR upper limit: the highest TPR reached by any phase at FPR = fpr_max.
    # Add a small margin (5 % of the range) so the curve isn't cut off.
    tpr_at_boundary = max(tpr_z.max() if len(tpr_z) else 0.0 for _, _, _, tpr_z, _ in masked)
    tpr_margin = max(0.02, tpr_at_boundary * 0.05)
    tpr_max = min(1.0, tpr_at_boundary + tpr_margin)

    fig, ax = plt.subplots(figsize=figsize)

    # Random-classifier diagonal (clipped to the zoomed window)
    ax.plot(
        [0, fpr_max], [0, fpr_max],
        color="0.75", linestyle="--", linewidth=1, zorder=1, label="Random",
    )

    for label, color, fpr_z, tpr_z, partial_auc in masked:
        ax.plot(
            fpr_z, tpr_z,
            color=color,
            linewidth=2.2,
            zorder=2,
            label=f"{label}  (pAUC = {partial_auc:.4f})",
        )

    ax.set_xlim(-0.001, fpr_max)
    ax.set_ylim(-0.02, tpr_max)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(
        f"Phase 1 vs Phase 2 ROC Curve Comparison (FPR ≤ {fpr_max})",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"plot_phase_roc_comparison_zoomed: saved plot to {output_path}")


_DEFAULT_CONFIG = (
    Path(__file__).parent / "src" / "anldq" / "configs" / "train_config" / "tranad.yml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPT histogram and ROC analysis")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to a TrainConfig YAML file. "
            "If not given, defaults to configs/ or the canonical src/ tranad.yml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.config is not None:
        config_path = args.config
    else:
        # Fall back to configs/ directory if it exists alongside this script,
        # otherwise use the canonical src/ default.
        configs_dir = Path(__file__).parent / "configs"
        config_path = _DEFAULT_CONFIG

    cfg = TrainConfig.from_yaml(str(config_path))
    print(f"Loaded config from: {config_path}")

    # run_id is set inside build_trainer (builder.py:128: f"{data_tag}_{model}").
    # Pre-populate it the same way so run_dir is correct in both branches.
    if not cfg.run_id:
        cfg.run_id = f"{cfg.data_tag}_TranAD"

    run_dir = cfg.io.get_experiment_dir(cfg.run_id)

    if RUN_TRAINING:
        train, val, test = load_dataset(
            "spt",
            data_variant="snr",
            label_from_snr=True,
            label_snr_threshold=LABEL_SNR_THRESHOLD,
            trim_test_timestamps=False,
        )

        # Copy the config into the run directory for reproducibility.
        run_dir.mkdir(parents=True, exist_ok=True)
        dest_config = run_dir / config_path.name
        shutil.copy2(config_path, dest_config)
        print(f"Copied config to: {dest_config}")

        build_trainer(cfg, dataset=train).train(train, val, test)

        # Best-model inference → errors.npy (used by standard / phase-2 ROC scan).
        infer = build_infer(cfg, load_best=True)
        infer.run(test, save_dir=run_dir)

        # Final-model inference → errors_phase2.npy (mirrors cli/main.py logic).
        infer_final = build_infer(cfg, load_best=False)
        infer_final.run(test, save_dir=run_dir)
        err_phase2 = infer_final._last_errors
        if err_phase2 is not None:
            np.save(run_dir / "errors_phase2.npy", err_phase2)
            print(f"[Phase2] errors_phase2.npy saved ({err_phase2.shape}) → {run_dir / 'errors_phase2.npy'}")

        phase2_epoch_dir = _save_phase2_checkpoint_errors(cfg, test, run_dir)

        np.save(run_dir / "test_data.npy", np.asarray(test.data))

        channel_labels = getattr(test, "channel_labels", None)
        if channel_labels is not None:
            np.save(run_dir / "test_labels.npy", np.asarray(channel_labels))
        if channel_labels is None:
            print("Skipping validation plot: test dataset does not provide channel_labels.")
            return
    else:
        print(f"Skipping training and inference. Loading existing results from {run_dir}")
        labels_path = run_dir / "test_labels.npy"
        if not labels_path.exists():
            print(f"Cannot plot: missing {labels_path}")
            return
        channel_labels = np.load(labels_path)

    # ------------------------------------------------------------------ #
    # Per-phase validation error histograms                              #
    # ------------------------------------------------------------------ #
    phase1_errors_path = run_dir / "errors_phase1.npy"
    phase2_errors_path = run_dir / "errors_phase2.npy"
    phase2_epoch_dir = run_dir / "phase2_checkpoint_errors"

    plot_validation_error_histogram(
        run_dir,
        channel_labels,
        threshold=LABEL_SNR_THRESHOLD,
        output_path=run_dir / f"{DATASET}_validation_error_histogram_phase1.png",
        errors_path=phase1_errors_path,
    )
    plot_validation_error_histogram(
        run_dir,
        channel_labels,
        threshold=LABEL_SNR_THRESHOLD,
        output_path=run_dir / f"{DATASET}_validation_error_histogram_phase2.png",
        errors_path=phase2_errors_path,
    )
    plot_validation_error_histogram_v2(
        run_dir,
        channel_labels,
        threshold=LABEL_SNR_THRESHOLD,
        output_path=run_dir / f"{DATASET}_validation_error_histogram_v2_phase1.png",
        errors_path=phase1_errors_path,
    )
    plot_validation_error_histogram_v2(
        run_dir,
        channel_labels,
        threshold=LABEL_SNR_THRESHOLD,
        output_path=run_dir / f"{DATASET}_validation_error_histogram_v2_phase2.png",
        errors_path=phase2_errors_path,
    )

    # ------------------------------------------------------------------ #
    # Phase-1 ROC scan — colourful single curve with colorbar            #
    # ------------------------------------------------------------------ #
    if phase1_errors_path.exists():
        compute_roc_scan(
            run_dir,
            channel_labels,
            threshold_min=1e-10,
            output_prefix=f"{DATASET}_roc_scan_phase1",
            errors_path=phase1_errors_path,
        )
        plot_roc_curve(
            run_dir,
            roc_prefix=f"{DATASET}_roc_scan_phase1",
            output_path=run_dir / f"{DATASET}_roc_curve_phase1.png",
            title="ROC Curve — Phase 1 (cooperative pre-training) TranAD",
        )
    else:
        print(
            f"Phase-1 errors not found at {phase1_errors_path}; "
            "skipping phase-1 ROC scan.\n"
            "(Run with adversarial training enabled to generate errors_phase1.npy.)"
        )

    # ------------------------------------------------------------------ #
    # Phase-2 ROC scan — reads errors_phase2.npy (final adversarial ckpt)#
    # ------------------------------------------------------------------ #
    if phase2_errors_path.exists():
        compute_roc_scan(
            run_dir,
            channel_labels,
            threshold_min=1e-10,
            output_prefix=f"{DATASET}_roc_scan_phase2",
            errors_path=phase2_errors_path,
        )
    else:
        print(
            f"Phase-2 errors not found at {phase2_errors_path}; "
            "skipping phase-2 ROC scan."
        )

    if phase2_epoch_dir.exists():
        for errors_path in sorted(phase2_epoch_dir.glob("errors_phase2_epoch_[0-9][0-9][0-9][0-9].npy")):
            epoch_suffix = errors_path.stem.rsplit("_", 1)[-1]
            roc_prefix = f"{DATASET}_roc_scan_phase2_epoch_{epoch_suffix}"
            compute_roc_scan(
                phase2_epoch_dir,
                channel_labels,
                threshold_min=1e-10,
                output_prefix=roc_prefix,
                errors_path=errors_path,
            )
            plot_roc_curve(
                phase2_epoch_dir,
                roc_prefix=roc_prefix,
                output_path=phase2_epoch_dir / f"{DATASET}_roc_curve_phase2_epoch_{epoch_suffix}.png",
                title=f"ROC Curve — Phase 2 TranAD (epoch {int(epoch_suffix)})",
            )
    else:
        print(
            f"Phase-2 checkpoint error directory not found at {phase2_epoch_dir}; "
            "skipping per-checkpoint phase-2 ROC scans."
        )

    # ------------------------------------------------------------------ #
    # Overlay: phase-1 (blue) vs phase-2 (red), no colorbar              #
    # ------------------------------------------------------------------ #
    plot_phase_roc_comparison(
        run_dir,
        phase1_prefix=f"{DATASET}_roc_scan_phase1",
        phase2_prefix=f"{DATASET}_roc_scan_phase2",
        output_path=run_dir / f"{DATASET}_roc_phase_comparison.png",
    )

    # ------------------------------------------------------------------ #
    # Zoomed overlay: FPR 0 → 0.05 (high-threshold region)               #
    # ------------------------------------------------------------------ #
    plot_phase_roc_comparison_zoomed(
        run_dir,
        phase1_prefix=f"{DATASET}_roc_scan_phase1",
        phase2_prefix=f"{DATASET}_roc_scan_phase2",
        output_path=run_dir / f"{DATASET}_roc_phase_comparison_zoom.png",
    )


if __name__ == "__main__":
    main()
