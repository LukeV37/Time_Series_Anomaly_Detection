"""Minimal TranAD training loop."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_training_config
from .data import create_data_loaders
from .models import TranAD


def train_tranad(
    npz_path: str | Path,
    *,
    window_size: int = 10,
    batch_size: int = 32,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    device: str | None = None,
) -> tuple[TranAD, dict[str, float]]:
    """Train a minimal TranAD model from a preprocessing ``.npz`` file."""
    train_loader, val_loader, test_loader, metadata = create_data_loaders(
        npz_path,
        window_size=window_size,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_workers=num_workers,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    last_train_loss = 0.0
    last_val_loss = 0.0
    for _epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for src, tgt in train_loader:
            src = src.to(resolved_device)
            tgt = tgt.to(resolved_device)
            optimizer.zero_grad()
            x1, x2 = model(src, tgt)
            loss = F.mse_loss(x1, tgt) + F.mse_loss(x2, tgt)
            loss.backward()
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
                loss = F.mse_loss(x1, tgt) + F.mse_loss(x2, tgt)
                val_loss_sum += float(loss.item())
                val_batches += 1
        last_val_loss = val_loss_sum / max(val_batches, 1)

    metrics = {
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "input_dims": float(input_dims),
        "train_windows": float(len(train_loader.dataset)),
        "val_windows": float(len(val_loader.dataset)),
        "years_count": float(len(metadata.get("years", []))),
    }
    return model, metrics


def train_tranad_from_config(config_path: str | Path) -> tuple[TranAD, dict[str, float], str | None]:
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

    model, metrics = train_tranad(
        config["input"]["npz_path"],
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
        device=training_config.get("device"),
    )

    checkpoint_path = output_config.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = str(checkpoint_path)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        metrics = dict(metrics)
        metrics["checkpoint_path"] = checkpoint_path

    return model, metrics, checkpoint_path
