"""Hand-written Mamba block for window-sequence classification.

Architecture:
    Mamba-1 base  (Gu & Dao, 2023)
      - selective SSM: per-timestep Δ, B, C projected from input
      - depthwise 1D causal convolution
      - SiLU gating + residual projection
    Mamba-3 upgrades  (ICLR 2026)
      - COMPLEX-VALUED A  : A has Re<0 (decay) and Im free (oscillation),
        equivalent to a data-dependent rotary embedding — richer state
        tracking than Mamba-2's real-diagonal A.
      - TRAPEZOIDAL DISCRETIZATION : replaces Euler's A_bar = exp(ΔA)
        with (I + Δ/2 A)/(I - Δ/2 A), better numerical behavior at large Δ.
    Skipped
      - MIMO SSM : our (d_inner × d_state) is too small to benefit.

Bidirectional wrapper concatenates forward + time-reversed passes so each
window can use both past and future context for classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaHybridBlock(nn.Module):
    """Selective SSM block with Mamba-3 complex-A + trapezoidal discretization."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # depthwise causal 1D conv
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner,
        )

        # selective Δ, B, C from x_inner
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        with torch.no_grad():
            # inverse-softplus(0.1) ≈ −2.25; yields Δ ~ 0.1 at init
            self.dt_proj.bias.fill_(-2.2)

        # --- Mamba-3 complex A: Re(A) = -exp(·) enforces decay; Im(A) free ---
        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_real_log = nn.Parameter(a_init.expand(self.d_inner, -1).clone())
        self.A_imag = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_model) -> (B, T, D_model)"""
        B, T, _ = x.shape

        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # causal 1D conv across time
        xt = x_in.transpose(1, 2)  # (B, d_inner, T)
        xt = self.conv1d(xt)[:, :, :T]
        xt = F.silu(xt)
        x_in = xt.transpose(1, 2)  # (B, T, d_inner)

        sel = self.x_proj(x_in)
        B_sel, C_sel, dt_raw = sel.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)

        # --- Mamba-3: trapezoidal discretization with complex A ---
        A_real = -torch.exp(self.A_real_log)  # (d_inner, d_state)
        A_imag = self.A_imag
        half = 0.5 * dt.unsqueeze(-1)  # (B, T, d_inner, 1)

        hdtA_re = half * A_real
        hdtA_im = half * A_imag

        # numer = 1 + Δ/2 · A ;  denom = 1 − Δ/2 · A  (complex)
        numer_re = 1.0 + hdtA_re
        numer_im = hdtA_im
        denom_re = 1.0 - hdtA_re
        denom_im = -hdtA_im
        denom_mag2 = denom_re ** 2 + denom_im ** 2

        # A_bar = numer / denom  (complex division)
        A_bar_re = (numer_re * denom_re + numer_im * denom_im) / denom_mag2
        A_bar_im = (numer_im * denom_re - numer_re * denom_im) / denom_mag2

        # B_bar = Δ / denom  (complex)
        dt_full = dt.unsqueeze(-1)
        B_bar_re = (dt_full * denom_re) / denom_mag2
        B_bar_im = (-dt_full * denom_im) / denom_mag2

        # sequential complex scan
        h_re = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        h_im = torch.zeros_like(h_re)
        y_out = torch.zeros(B, T, self.d_inner, device=x.device, dtype=x.dtype)

        for t in range(T):
            Bt = B_sel[:, t, :].unsqueeze(1)         # (B, 1, d_state)
            xt_val = x_in[:, t, :].unsqueeze(-1)     # (B, d_inner, 1)
            Bx = xt_val * Bt                          # (B, d_inner, d_state) real

            A_re_t = A_bar_re[:, t]
            A_im_t = A_bar_im[:, t]
            B_re_t = B_bar_re[:, t]
            B_im_t = B_bar_im[:, t]

            new_re = A_re_t * h_re - A_im_t * h_im + B_re_t * Bx
            new_im = A_re_t * h_im + A_im_t * h_re + B_im_t * Bx
            h_re, h_im = new_re, new_im

            Ct = C_sel[:, t, :].unsqueeze(1)          # (B, 1, d_state)
            y_out[:, t, :] = (h_re * Ct).sum(dim=-1)  # Re(C·h) summed over d_state

        y_out = y_out + self.D * x_in
        y_out = y_out * F.silu(gate)
        return self.out_proj(y_out)


class BidirectionalMamba(nn.Module):
    """Forward Mamba + time-reversed Mamba, summed (so each window sees both directions)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.fwd = MambaHybridBlock(d_model, d_state, d_conv, expand)
        self.bwd = MambaHybridBlock(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.fwd(x)
        y_bwd = self.bwd(torch.flip(x, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        return y_fwd + y_bwd


class WindowSequenceClassifier(nn.Module):
    """(B, T, D_features) -> per-window class logits (B, T, n_classes)."""

    def __init__(
        self,
        d_features: int,
        d_model: int = 64,
        n_blocks: int = 2,
        d_state: int = 16,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(d_features, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [BidirectionalMamba(d_model, d_state=d_state) for _ in range(n_blocks)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_in(self.embed(x))
        for blk, ln in zip(self.blocks, self.norms):
            h = h + self.dropout(blk(ln(h)))
        return self.head(h)
