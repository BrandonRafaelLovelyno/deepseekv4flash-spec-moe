"""Predictor architectures for the single-variant routing prototype.

All architectures map an activation vector to the exact score space of the
reference router, ``sqrt(softplus(logits))``. No layer carries a bias: the
target is the *pre*-bias score and DeepSeek's gate projection has no bias.

Imported only inside the remote training function, so the local entrypoint
never needs torch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Reference score activation: ``sqrt(softplus(logits))``."""
    return F.softplus(logits).sqrt()


class LowRankAdapter(nn.Module):
    """``sqrt(softplus(ABx))``: a rank-``r`` reparameterization of the router."""

    def __init__(self, d_in: int, n_experts: int, rank: int = 128) -> None:
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scores_from_logits(self.up(self.down(x)))


class SwiGLUMLP(nn.Module):
    """Two-layer MLP: ``sqrt(softplus(out(silu(gate(x)) * up(x))))``."""

    def __init__(self, d_in: int, n_experts: int, hidden: int = 4096) -> None:
        super().__init__()
        self.gate = nn.Linear(d_in, hidden, bias=False)
        self.up = nn.Linear(d_in, hidden, bias=False)
        self.out = nn.Linear(hidden, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.up(x)
        return scores_from_logits(self.out(h))


class MLP(nn.Module):
    """Plain two-layer MLP: ``sqrt(softplus(down(gelu(up(x)))))``.

    Unlike ``SwiGLUMLP`` the hidden layer is a single GELU projection, not a
    gated product.
    """

    def __init__(self, d_in: int, n_experts: int, hidden: int = 4096) -> None:
        super().__init__()
        self.up = nn.Linear(d_in, hidden, bias=False)
        self.down = nn.Linear(hidden, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return scores_from_logits(self.down(F.gelu(self.up(x))))


def build_model(
    arch: str,
    d_in: int,
    n_experts: int,
    rank: int = 128,
    hidden: int = 4096,
) -> nn.Module:
    """Construct one predictor by name."""
    if arch == "lowrank":
        return LowRankAdapter(d_in, n_experts, rank=rank)
    if arch == "swiglu":
        return SwiGLUMLP(d_in, n_experts, hidden=hidden)
    if arch == "mlp":
        return MLP(d_in, n_experts, hidden=hidden)
    raise ValueError(f"unknown arch: {arch!r}")
