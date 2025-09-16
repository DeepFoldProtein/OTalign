import torch
import torch.nn.functional as F


def pairwise_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [m, d], y: [n, d]
    x2 = (x * x).sum(-1, keepdim=True)  # [m, 1]
    y2 = (y * y).sum(-1, keepdim=True).T  # [1, n]
    xy = x @ y.T  # [m, n]
    return (x2 - 2 * xy + y2).clamp_min(0.0)


def pairwise_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = F.normalize(x, dim=-1, eps=eps)
    y = F.normalize(y, dim=-1, eps=eps)
    # convert similarity to cost
    sim = x @ y.T
    return 1.0 - sim
