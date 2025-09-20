import torch

from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn

from .cost import pairwise_cosine


@torch.no_grad()
def align_embeddings(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    lambda1: float,
    lambda2: float,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    cost_fn=pairwise_cosine,
    **kwargs,
) -> dict:
    # x: [m, d], y: [n, d]
    c = cost_fn(x, y)  # [m, n]
    m, n = c.shape
    if a is None:
        a = torch.full((m,), 1.0 / m, dtype=c.dtype, device=c.device)
    if b is None:
        b = torch.full((n,), 1.0 / n, dtype=c.dtype, device=c.device)
    plan, u, v = unbalanced_sinkhorn(
        c,
        a,
        b,
        num_iter=kwargs.get("num_iter", 1000),
        reg=epsilon,
        lambda1=lambda1,
        lambda2=lambda2,
        mask_a=kwargs.get("mask_a"),
        mask_b=kwargs.get("mask_b"),
        tol=kwargs.get("tol", 1e-4),
        damp=kwargs.get("damp", 1e-6),
        gauge=kwargs.get("gauge", "last_beta"),
        stabilize=kwargs.get("stabilize", False),
    )
    return {"plan": plan, "u": u, "v": v, "cost": (plan * c).sum((-2, -1))}
