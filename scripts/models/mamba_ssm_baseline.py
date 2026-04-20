"""Mamba-1 reference baseline using Tri Dao's fused CUDA selective_scan kernel.

Ablation reference for the hand-written Mamba-3 hybrid in `mamba_hybrid.py`.
This model keeps the `WindowSequenceClassifier` interface (same constructor
signature and (B, T, D_features) -> (B, T, n_classes) forward) but swaps in
the official `mamba_ssm.Mamba` block, which is vanilla Mamba-1: real-diagonal
A and Euler discretization, executed by the fused CUDA kernel.

Contrast:
    mamba_hybrid.py  : hand-written Mamba-1 base + Mamba-3 upgrades
                       (complex-valued A, trapezoidal discretization), bidirectional
    this file        : fused-kernel Mamba-1 reference (real A, Euler), bidirectional

Seeds are set externally in the trainer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm import Mamba


class BidirectionalMambaSSM(nn.Module):
    """Forward + time-reversed mamba_ssm.Mamba blocks, summed."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        return y_fwd + y_bwd


class MambaSSMv1Classifier(nn.Module):
    """(B, T, D_features) -> per-window class logits (B, T, n_classes).

    Requires CUDA: mamba_ssm.Mamba ships a fused selective_scan kernel with
    no CPU fallback. Constructing this model on a CPU-only host would only
    fail later inside the kernel with a confusing error.
    """

    def __init__(
        self,
        d_features: int,
        d_model: int = 64,
        n_blocks: int = 2,
        d_state: int = 16,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MambaSSMv1Classifier requires CUDA; mamba_ssm has no CPU kernel. "
                "Re-run with --device cuda, or drop mamba_ssm_v1 from --models."
            )
        super().__init__()
        self.embed = nn.Linear(d_features, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [BidirectionalMambaSSM(d_model, d_state=d_state) for _ in range(n_blocks)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(self.embed(x))
        for blk, ln in zip(self.blocks, self.norms):
            h = h + self.dropout(blk(ln(h)))
        return self.head(h)
