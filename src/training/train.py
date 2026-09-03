"""Minimal TranAD training loop."""

from __future__ import annotations

from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import load_training_config
from .data import create_data_loaders
from .models import TranAD


def _copy_config_to_output_dir(config_path: str | Path, output_dir: str | Path) -> str:
    """Copy the training config next to the generated artifacts for reproducibility."""
    source = Path(config_path)
    destination = Path(output_dir) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _save_loss_curve(
    output_path: str | Path,
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    """Save a log-scale train/validation loss curve."""
    epochs = np.arange(1, len(train_losses) + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, train_losses, label="Train Loss", color="blue", marker=".", linewidth=1.0)
    ax.semilogy(epochs, val_losses, label="Val Loss", color="green", marker="o", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("TranAD Loss Curve")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(loc="best")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _collect_reconstruction_errors(
    model: TranAD,
    data_loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    channel_count: int,
) -> np.ndarray:
    """Run inference and collect per-timestep, per-channel squared reconstruction errors."""
    model.eval()
    errors = []
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            _x1, x2 = model(src, tgt)
            batch_errors = (x2 - tgt).pow(2).reshape(tgt.shape[0], channel_count, -1)
            errors.append(batch_errors.cpu().numpy())
    return np.concatenate(errors, axis=0)


def train_tranad(
    train_npz_path: str | Path,
    *,
    test_npz_path: str | Path | None = None,
    window_size: int = 10,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    final_learning_rate: float | None = None,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    weight_decay: float = 1.0e-5,
    max_grad_norm: float = 1.0,
    device: str | None = None,
    scaling: str = "standard",
    loss_curve_path: str | Path | None = None,
) -> tuple[TranAD, dict[str, float]]:
    """Train a minimal TranAD model from preprocessing ``.npz`` files."""
    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        train_npz_path,
        test_npz_path=test_npz_path,
        window_size=window_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=num_workers,
        scaling=scaling,
    )
    _ = test_loader

    input_dims = train_loader.dataset.tensors[0].shape[-1]
    model = TranAD(
        input_dims=input_dims,
        n_window=window_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if final_learning_rate is None:
        final_learning_rate = learning_rate
    if final_learning_rate <= 0.0:
        raise ValueError("final_learning_rate must be positive.")
    scheduler = None
    if epochs > 1 and final_learning_rate != learning_rate:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=final_learning_rate / learning_rate,
            total_iters=epochs - 1,
        )

    last_train_loss = 0.0
    last_val_loss = 0.0
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        current_learning_rate = optimizer.param_groups[0]["lr"]

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for src, tgt in train_loader:
            src = src.to(resolved_device)
            tgt = tgt.to(resolved_device)
            optimizer.zero_grad()
            x1, x2 = model(src, tgt)
            phase_one_weight = 1.0 / (epoch + 1)
            loss = (
                phase_one_weight * F.mse_loss(x1, tgt)
                + (1.0 - phase_one_weight) * F.mse_loss(x2, tgt)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch + 1}, batch={train_batches + 1}. "
                    "Check dataset values or lower the learning rate."
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch + 1}, batch={train_batches + 1}."
                )
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_batches += 1
        last_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(resolved_device)
                tgt = tgt.to(resolved_device)
                x1, x2 = model(src, tgt)
                phase_one_weight = 1.0 / (epoch + 1)
                loss = (
                    phase_one_weight * F.mse_loss(x1, tgt)
                    + (1.0 - phase_one_weight) * F.mse_loss(x2, tgt)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch={epoch + 1}, batch={val_batches + 1}."
                    )
                val_loss_sum += float(loss.item())
                val_batches += 1
        last_val_loss = val_loss_sum / max(val_batches, 1)
        train_losses.append(last_train_loss)
        val_losses.append(last_val_loss)

        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={last_train_loss:.6f} "
            f"val_loss={last_val_loss:.6f} "
            f"lr={current_learning_rate:.3g} "
            f"seconds={epoch_seconds:.2f}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()

    split_source = metadata.get("split_source")
    if isinstance(split_source, torch.Tensor):
        split_source = split_source.item()
    elif hasattr(split_source, "item"):
        split_source = split_source.item()

    if loss_curve_path:
        _save_loss_curve(loss_curve_path, train_losses, val_losses)

    metrics = {
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "learning_rate": float(learning_rate),
        "final_learning_rate": float(final_learning_rate),
        "input_dims": float(input_dims),
        "train_windows": float(len(train_loader.dataset)),
        "val_windows": float(len(val_loader.dataset)),
        "years_count": float(len(metadata.get("years", []))),
        "split_source": split_source,
    }
    return model, metrics


def train_tranad_from_config(
    config_path: str | Path,
) -> tuple[TranAD, dict[str, float], str | None, str | None]:
    """Train TranAD from a nested YAML config."""
    config = load_training_config(config_path)
    model_type = config["model"]["type"]
    if model_type != "tranad":
        raise ValueError(f"Unsupported model type {model_type!r}. Expected 'tranad'.")

    split_config = config.get("split", {})
    loader_config = config.get("loader", {})
    model_params = dict(config.get("model", {}).get("params", {}))
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    input_config = config["input"]
    train_npz_path = input_config.get("train_npz_path") or input_config.get("npz_path")
    test_npz_path = input_config.get("test_npz_path")
    if train_npz_path is None:
        raise ValueError("Training config requires input.train_npz_path or legacy input.npz_path.")

    loss_curve_path = output_config.get("loss_curve_path")
    if not loss_curve_path and output_config.get("test_errors_path"):
        loss_curve_path = str(Path(output_config["test_errors_path"]).with_name("loss_curve.png"))
    elif not loss_curve_path and output_config.get("checkpoint_path"):
        loss_curve_path = str(Path(output_config["checkpoint_path"]).with_name("loss_curve.png"))

    model, metrics = train_tranad(
        train_npz_path,
        test_npz_path=test_npz_path,
        window_size=int(model_params.get("window_size", 10)),
        d_model=int(model_params.get("d_model", 128)),
        nhead=int(model_params.get("nhead", 8)),
        num_layers=int(model_params.get("num_layers", 2)),
        dim_feedforward=int(model_params.get("dim_feedforward", 256)),
        dropout=float(model_params.get("dropout", 0.1)),
        batch_size=int(loader_config.get("batch_size", 32)),
        num_workers=int(loader_config.get("num_workers", 0)),
        train_ratio=float(split_config.get("train_ratio", 0.6)),
        val_ratio=float(split_config.get("val_ratio", 0.2)),
        epochs=int(training_config.get("epochs", 5)),
        learning_rate=float(training_config.get("learning_rate", 1e-4)),
        final_learning_rate=(
            float(training_config["final_learning_rate"])
            if "final_learning_rate" in training_config
            else None
        ),
        weight_decay=float(training_config.get("weight_decay", 1.0e-5)),
        max_grad_norm=float(training_config.get("max_grad_norm", 1.0)),
        device=training_config.get("device"),
        scaling=str(training_config.get("scaling", "standard")),
        loss_curve_path=loss_curve_path,
    )

    if loss_curve_path:
        metrics = dict(metrics)
        metrics["loss_curve_path"] = str(loss_curve_path)

    output_dir = None
    for candidate in (loss_curve_path, output_config.get("test_errors_path"), output_config.get("checkpoint_path")):
        if candidate:
            output_dir = str(Path(candidate).parent)
            break
    if output_dir:
        metrics = dict(metrics)
        metrics["config_copy_path"] = _copy_config_to_output_dir(config_path, output_dir)

    checkpoint_path = output_config.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = str(checkpoint_path)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        metrics = dict(metrics)
        metrics["checkpoint_path"] = checkpoint_path

    test_errors_path = output_config.get("test_errors_path")
    if test_errors_path:
        test_errors_path = str(test_errors_path)
        loader_config = config.get("loader", {})
        model_params = dict(config.get("model", {}).get("params", {}))
        split_config = config.get("split", {})
        training_config = config.get("training", {})
        _train_loader, _val_loader, test_loader, inference_metadata = create_data_loaders(
            train_npz_path,
            test_npz_path=test_npz_path,
            window_size=int(model_params.get("window_size", 10)),
            batch_size=int(loader_config.get("batch_size", 32)),
            train_ratio=float(split_config.get("train_ratio", 0.6)),
            val_ratio=float(split_config.get("val_ratio", 0.2)),
            num_workers=int(loader_config.get("num_workers", 0)),
            scaling=str(training_config.get("scaling", "standard")),
        )
        resolved_device = torch.device(
            training_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        channel_names = inference_metadata.get("channel_names")
        channel_count = int(len(channel_names)) if channel_names is not None else int(metrics["input_dims"])
        test_errors = _collect_reconstruction_errors(
            model,
            test_loader,
            device=resolved_device,
            channel_count=channel_count,
        )
        Path(test_errors_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(test_errors_path, test_errors)
        metrics = dict(metrics)
        metrics["test_errors_path"] = test_errors_path
        metrics["test_windows"] = float(test_errors.shape[0])

    return model, metrics, checkpoint_path, test_errors_path if test_errors_path else None
