import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import anldq.deep_learning  # registers models in the registry
from anldq.configs import TrainConfig
from anldq.datasets import load_dataset
from anldq.infer import build_infer

DATASET = "spt"
LABEL_SNR_THRESHOLD = 20.0
_DEFAULT_CONFIG = Path(__file__).parent / "src" / "anldq" / "configs" / "train_config" / "tranad.yml"


def compute_roc_scan(
    run_dir: Path,
    labels: np.ndarray,
    *,
    threshold_min: float = 1e-3,
    n_points: int = 500,
    output_prefix: str = "roc_scan",
    errors_path: Path,
) -> None:
    if not errors_path.exists():
        print(f"compute_roc_scan: missing errors file at {errors_path}")
        return

    errors = np.load(errors_path)
    labels = np.asarray(labels)
    if errors.shape != labels.shape:
        print(
            f"compute_roc_scan: labels shape {labels.shape} does not match "
            f"errors shape {errors.shape} - aborting."
        )
        return

    errors_flat = errors.ravel().astype(float)
    labels_flat = labels.ravel().astype(int)
    finite_mask = np.isfinite(errors_flat) & (errors_flat > 0)
    errors_flat = errors_flat[finite_mask]
    labels_flat = labels_flat[finite_mask]

    if errors_flat.size == 0:
        print("compute_roc_scan: no positive finite reconstruction errors - aborting.")
        return

    pos_mask = labels_flat == 1
    neg_mask = labels_flat == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        print(
            f"compute_roc_scan: need both classes present (n_pos={n_pos}, n_neg={n_neg}) - aborting."
        )
        return

    threshold_max = errors_flat.max()
    if threshold_min >= threshold_max:
        print(
            f"compute_roc_scan: threshold_min ({threshold_min:.3g}) >= "
            f"max error ({threshold_max:.3g}) - aborting."
        )
        return

    thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), n_points)
    tpr = np.empty(n_points, dtype=float)
    fpr = np.empty(n_points, dtype=float)
    for i, threshold in enumerate(thresholds):
        predicted_pos = errors_flat >= threshold
        tpr[i] = predicted_pos[pos_mask].sum() / n_pos
        fpr[i] = predicted_pos[neg_mask].sum() / n_neg

    npz_path = run_dir / f"{output_prefix}.npz"
    np.savez(npz_path, thresholds=thresholds, tpr=tpr, fpr=fpr)
    csv_path = run_dir / f"{output_prefix}.csv"
    np.savetxt(
        csv_path,
        np.column_stack([thresholds, tpr, fpr]),
        delimiter=",",
        header="threshold,tpr,fpr",
        comments="",
        fmt="%.8g",
    )
    print(f"compute_roc_scan: saved arrays to {npz_path}")
    print(f"compute_roc_scan: saved CSV to {csv_path}")


def plot_phase2_roc_evolution(
    run_dir: Path,
    roc_series: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    output_path: Path,
    title: str,
    fpr_max: float | None = None,
    cmap: str = "viridis",
) -> None:
    if not roc_series:
        print("plot_phase2_roc_evolution: no ROC series provided; skipping.")
        return

    epochs = np.asarray([epoch for epoch, _, _ in roc_series], dtype=float)
    norm = plt.Normalize(vmin=epochs.min(), vmax=epochs.max())
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(7, 6))
    diagonal_max = 1.0 if fpr_max is None else fpr_max
    ax.plot([0, diagonal_max], [0, diagonal_max], color="0.75", linestyle="--", linewidth=1, zorder=1)

    visible_tpr_max = 0.0
    for epoch, fpr, tpr in roc_series:
        mask = np.ones_like(fpr, dtype=bool)
        if fpr_max is not None:
            mask = fpr <= fpr_max
        fpr_plot = fpr[mask]
        tpr_plot = tpr[mask]
        if not len(fpr_plot):
            continue
        visible_tpr_max = max(visible_tpr_max, float(np.max(tpr_plot)))
        ax.plot(
            fpr_plot,
            tpr_plot,
            color=cmap_obj(norm(epoch)),
            linewidth=1.8,
            alpha=0.95,
            zorder=2,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Checkpoint epoch", fontsize=10)

    x_max = 1.0 if fpr_max is None else fpr_max
    y_max = 1.02 if fpr_max is None else min(1.0, max(0.05, visible_tpr_max * 1.05))
    ax.set_xlim(-0.02 if fpr_max is None else -0.001, x_max)
    ax.set_ylim(-0.02, y_max)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"plot_phase2_roc_evolution: saved plot to {output_path}")


def _load_loss_csv(loss_csv_path: Path):
    if not loss_csv_path.exists():
        print(f"Skipping loss plot: missing CSV at {loss_csv_path}")
        return None

    data = np.genfromtxt(loss_csv_path, delimiter=",", names=True)
    if data.size == 0:
        print(f"Skipping loss plot: empty CSV at {loss_csv_path}")
        return None

    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data



def plot_loss_from_csv(
    loss_csv_path: Path,
    output_path: Path,
    title: str,
    *,
    epoch_min: int | None = None,
    epoch_max: int | None = None,
) -> None:
    data = _load_loss_csv(loss_csv_path)
    if data is None:
        return

    epochs_all = np.asarray(data["epoch"], dtype=float)
    train_key = "train_eval_loss" if "train_eval_loss" in data.dtype.names else "train_loss"
    train_vals_all = np.asarray(data[train_key], dtype=float)
    val_vals_all = np.asarray(data["val_loss"], dtype=float) if "val_loss" in data.dtype.names else None
    lr_vals_all = np.asarray(data["learning_rate"], dtype=float) if "learning_rate" in data.dtype.names else None

    epoch_window = np.ones_like(epochs_all, dtype=bool)
    if epoch_min is not None:
        epoch_window &= epochs_all >= epoch_min
    if epoch_max is not None:
        epoch_window &= epochs_all <= epoch_max

    train_mask = np.isfinite(train_vals_all) & epoch_window
    train_epochs = epochs_all[train_mask]
    train_plot = train_vals_all[train_mask]
    if not len(train_plot):
        print(f"Skipping loss plot: no train points in requested epoch window for {output_path}")
        return

    val_epochs = np.array([], dtype=float)
    val_plot = np.array([], dtype=float)
    if val_vals_all is not None:
        val_mask = np.isfinite(val_vals_all) & epoch_window
        val_epochs = epochs_all[val_mask]
        val_plot = val_vals_all[val_mask]

    ratio_epochs = []
    ratios = []
    train_by_epoch = {int(epoch): loss for epoch, loss in zip(train_epochs, train_plot)}
    for epoch, loss_val in zip(val_epochs, val_plot):
        loss_train = train_by_epoch.get(int(epoch))
        if loss_train is None or not np.isfinite(loss_train) or loss_train == 0:
            continue
        ratio_epochs.append(epoch)
        ratios.append(loss_val / loss_train)

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax.semilogy(
        train_epochs,
        train_plot,
        label="Train Eval Loss" if train_key == "train_eval_loss" else "Train Loss",
        color="blue",
        marker=".",
        linewidth=1.0,
    )
    if len(val_plot):
        ax.semilogy(val_epochs, val_plot, label="Val Loss", color="green", marker="o", linewidth=1.0)
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")

    if lr_vals_all is not None:
        lr_mask = np.isfinite(lr_vals_all) & epoch_window
        lr_plot = lr_vals_all[lr_mask]
        if len(lr_plot):
            initial_lr = lr_plot[0]
            final_lr = lr_plot[-1]
            lr_text = f"Learning Rate: {initial_lr:.2e}" if initial_lr == final_lr else f"Learning Rate: {initial_lr:.2e} -> {final_lr:.2e}"
            ax.text(
                0.98,
                0.98,
                lr_text,
                transform=ax.transAxes,
                fontsize=10,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            )

    if ratio_epochs:
        rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        rax.plot(ratio_epochs, ratios, color="purple", marker="o", linewidth=1.0)
    else:
        rax.text(0.5, 0.5, "No overlapping val/train points", ha="center", va="center", transform=rax.transAxes)
    rax.set_xlabel("Epochs")
    rax.set_ylabel("Val/Train")
    rax.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {output_path}")


def load_spt_test_data():
    _, _, test = load_dataset(
        "spt",
        data_variant="snr",
        label_from_snr=True,
        label_snr_threshold=LABEL_SNR_THRESHOLD,
        trim_test_timestamps=False,
    )
    channel_labels = getattr(test, "channel_labels", None)
    if channel_labels is None:
        raise RuntimeError("SPT test dataset does not provide channel_labels.")
    return test, np.asarray(channel_labels)


def get_phase2_checkpoint_epochs(cfg: TrainConfig) -> list[int]:
    adv_start_epoch = getattr(getattr(cfg, "tranad_adv", None), "adv_start_epoch", None)
    if adv_start_epoch is None:
        return []

    ckpt_dir = cfg.io.get_checkpoint_dir(cfg.run_id)
    epochs: list[int] = []
    for ckpt_path in sorted(ckpt_dir.glob("model_[0-9][0-9][0-9][0-9].ckpt")):
        try:
            epoch = int(ckpt_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if epoch >= adv_start_epoch:
            epochs.append(epoch)
    return epochs


def save_phase2_checkpoint_errors(cfg: TrainConfig, test_ds, run_dir: Path) -> tuple[Path, list[tuple[int, Path]]]:
    output_dir = run_dir / "phase2_checkpoint_errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = get_phase2_checkpoint_epochs(cfg)
    if not epochs:
        print("[Phase2] No saved phase-2 checkpoints found for per-epoch inference.")
        return output_dir, []

    saved_errors: list[tuple[int, Path]] = []
    for epoch in epochs:
        infer = build_infer(cfg, load_best=False, load_epoch=epoch)
        infer.run(test_ds, save_dir=output_dir)
        errors = infer._last_errors
        if errors is None:
            print(f"[Phase2] No errors returned for epoch {epoch:04d}; skipping save.")
            continue
        errors_path = output_dir / f"errors_phase2_epoch_{epoch:04d}.npy"
        np.save(errors_path, errors)
        saved_errors.append((epoch, errors_path))
        print(f"[Phase2] saved {errors_path.name} ({errors.shape}) -> {errors_path}")
    return output_dir, saved_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPT post-training inference and plotting")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        metavar="PATH",
        help="Path to a TrainConfig YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(str(args.config))
    print(f"Loaded config from: {args.config}")

    if not cfg.run_id:
        cfg.run_id = f"{cfg.data_tag}_TranAD"
    run_dir = cfg.io.get_experiment_dir(cfg.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    loss_csv_path = run_dir / "loss_curve.csv"
    plot_loss_from_csv(
        loss_csv_path,
        run_dir / "loss_curve_from_csv.png",
        "Loss curve from loss_curve.csv",
    )

    adv_start_epoch = getattr(getattr(cfg, "tranad_adv", None), "adv_start_epoch", None)
    if adv_start_epoch is not None:
        plot_loss_from_csv(
            loss_csv_path,
            run_dir / "loss_curve_phase1_from_csv.png",
            f"Loss curve from loss_curve.csv - phase 1 (epochs 1-{adv_start_epoch - 1})",
            epoch_max=adv_start_epoch - 1,
        )
        plot_loss_from_csv(
            loss_csv_path,
            run_dir / "loss_curve_phase2_from_csv.png",
            f"Loss curve from loss_curve.csv - phase 2 (epochs {adv_start_epoch}+)",
            epoch_min=adv_start_epoch,
        )

    test_ds, channel_labels = load_spt_test_data()
    np.save(run_dir / "test_labels.npy", channel_labels)

    phase2_epoch_dir, saved_errors = save_phase2_checkpoint_errors(cfg, test_ds, run_dir)

    roc_series: list[tuple[int, np.ndarray, np.ndarray]] = []
    for epoch, errors_path in saved_errors:
        roc_prefix = f"{DATASET}_roc_scan_phase2_epoch_{epoch:04d}"
        compute_roc_scan(
            phase2_epoch_dir,
            channel_labels,
            threshold_min=1e-10,
            output_prefix=roc_prefix,
            errors_path=errors_path,
        )
        roc_data = np.load(phase2_epoch_dir / f"{roc_prefix}.npz")
        roc_series.append((epoch, roc_data["fpr"], roc_data["tpr"]))

    plot_phase2_roc_evolution(
        phase2_epoch_dir,
        roc_series,
        output_path=run_dir / f"{DATASET}_roc_phase2_epoch_gradient.png",
        title="Phase-2 ROC Evolution Across Checkpoints",
    )
    plot_phase2_roc_evolution(
        phase2_epoch_dir,
        roc_series,
        output_path=run_dir / f"{DATASET}_roc_phase2_epoch_gradient_zoom.png",
        title="Phase-2 ROC Evolution Across Checkpoints (FPR <= 0.05)",
        fpr_max=0.05,
    )


if __name__ == "__main__":
    main()
