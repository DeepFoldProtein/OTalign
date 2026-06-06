"""Pairwise residue cost functions for OT alignment.

Cost modes share a uniform, mask-aware signature so new ones (e.g. a learned
Mahalanobis metric) can be registered without touching the aligner:

    cost_fn(x, y, mask_x=None, mask_y=None, eps=...) -> cost[B, Lx, Ly]

`x`/`y` are `[B, L, D]`; masks are `[B, L]` (True = valid residue). A mode that
does not need masks simply ignores them. Select a mode by name via
:func:`get_cost_fn`; the registry is :data:`COST_MODES`.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F


CostFn = Callable[..., torch.Tensor]


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [m, d], y: [n, d]
    x2 = (x * x).sum(-1, keepdim=True)  # [m, 1]
    y2 = (y * y).sum(-1, keepdim=True).T  # [1, n]
    xy = x @ y.transpose(-1, -2)  # [m, n]
    return (x2 - 2 * xy + y2).clamp_min(0.0)


def pairwise_cosine(
    x: torch.Tensor,
    y: torch.Tensor,
    mask_x: Optional[torch.Tensor] = None,
    mask_y: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Plain cosine cost ``1 - cos``. Masks are ignored (cosine is per-residue)."""
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    sim = x @ y.transpose(-1, -2)
    return 1.0 - sim


def pairwise_cosine_centered(
    x: torch.Tensor,
    y: torch.Tensor,
    mask_x: Optional[torch.Tensor] = None,
    mask_y: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine cost after removing the shared mean direction of the residues.

    PLM residue embeddings are anisotropic: a dominant common component makes
    unrelated residues look similar and creates spurious "anchor"/hub matches.
    Subtracting the mean direction over all valid residues of *both* sequences
    (per batch element) before cosine de-emphasizes that hub component. Masks
    keep padding from biasing the mean.
    """
    if mask_x is None:
        mask_x = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
    if mask_y is None:
        mask_y = torch.ones(y.shape[:-1], dtype=torch.bool, device=y.device)
    mx = mask_x.unsqueeze(-1).to(x.dtype)
    my = mask_y.unsqueeze(-1).to(y.dtype)
    summed = (x * mx).sum(dim=-2) + (y * my).sum(dim=-2)  # [B, D]
    count = mask_x.sum(dim=-1, keepdim=True) + mask_y.sum(dim=-1, keepdim=True)  # [B, 1]
    mean = (summed / count.clamp_min(1.0)).unsqueeze(-2)  # [B, 1, D]
    x = F.normalize(x - mean, dim=-1, eps=eps)
    y = F.normalize(y - mean, dim=-1, eps=eps)
    sim = x @ y.transpose(-1, -2)
    return 1.0 - sim


# Registry of selectable cost modes. Add new metrics (e.g. "mahalanobis") here.
COST_MODES: dict[str, CostFn] = {
    "cosine": pairwise_cosine,
    "cosine_centered": pairwise_cosine_centered,
}


def get_cost_fn(mode: str) -> CostFn:
    """Resolve a cost mode name to its function."""
    if mode not in COST_MODES:
        raise ValueError(f"Unknown cost mode {mode!r}. Available: {sorted(COST_MODES)}")
    return COST_MODES[mode]
