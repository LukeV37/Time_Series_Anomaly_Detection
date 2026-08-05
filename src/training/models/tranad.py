"""Minimal standalone TranAD model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding with batch-first tensors."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TranAD(nn.Module):
    """Minimal two-phase TranAD for inputs ``src=(B, W, F)`` and ``tgt=(B, 1, F)``."""

    def __init__(
        self,
        input_dims: int,
        *,
        n_window: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_dims = int(input_dims)
        self.n_window = int(n_window)

        self.input_projection = nn.Linear(2 * self.input_dims, d_model)
        self.target_projection = nn.Linear(2 * self.input_dims, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=n_window)

        enc_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        dec_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder1 = TransformerDecoder(dec_layer, num_layers=num_layers)
        self.decoder2 = TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.input_dims)

    def _encode(self, src: Tensor, ctx: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
        merged = torch.cat((src, ctx), dim=-1) * math.sqrt(self.input_dims)
        memory = self.encoder(self.pos_encoder(self.input_projection(merged)))
        tgt_rep = self.target_projection(tgt.repeat(1, 1, 2))
        return tgt_rep, memory

    def forward(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
        ctx = torch.zeros_like(src)
        tgt_rep, memory = self._encode(src, ctx, tgt)
        x1 = self.output_proj(self.decoder1(tgt_rep, memory))

        ctx = (x1.repeat(1, src.size(1), 1) - src) ** 2
        tgt_rep, memory = self._encode(src, ctx, tgt)
        x2 = self.output_proj(self.decoder2(tgt_rep, memory))
        return x1, x2
