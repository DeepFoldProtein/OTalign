import torch
import torch.nn as nn

from otalign.align.cost import pairwise_cosine
from otalign.functional.sinkhorn_uot import unbalanced_sinkhorn


class SinkhornUOT(nn.Module):
    """
    A torch.nn.Module wrapper for the SinkhornUOT autograd function.

    This class computes a cost matrix between two sets of embeddings,
    then runs the unbalanced Sinkhorn algorithm to find an optimal
    transport plan.
    """

    def __init__(self, cost_fn=pairwise_cosine):
        """
        Initializes the SinkhornUOT aligner.

        Args:
            cost_fn: A function that takes two tensors (B, L, D) and
                     returns a cost matrix (B, L, L).
                     Defaults to pairwise_cosine.
        """
        super().__init__()
        self.cost_fn = cost_fn

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        num_iter: int,
        reg: float,
        lambda1: float,
        lambda2: float,
        u_init: torch.Tensor | None = None,
        v_init: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """
        Performs the forward pass of the alignment.

        Args:
            a (torch.Tensor): Embeddings for the first sequence (B, L_a, D).
            b (torch.Tensor): Embeddings for the second sequence (B, L_b, D).
            mask_a (torch.Tensor): Boolean mask for sequence a (B, L_a).
            mask_b (torch.Tensor): Boolean mask for sequence b (B, L_b).
            num_iter (int): Number of Sinkhorn iterations.
            reg (float): Entropic regularization strength.
            lambda1 (float): Marginal relaxation weight for sequence a.
            lambda2 (float): Marginal relaxation weight for sequence b.
            u_init (torch.Tensor | None): Initial dual variable for a.
            v_init (torch.Tensor | None): Initial dual variable for b.
            **kwargs: Additional arguments for the Sinkhorn algorithm.

        Returns:
            A dictionary containing the transport plan, dual variables,
            cost matrix, and marginals.
        """
        # Cost matrix
        cost_matrix = self.cost_fn(a, b)  # (B, L_a, L_b)

        # Marginals (uniform over valid lengths)
        mu = mask_a.float() / mask_a.sum(-1, keepdim=True).clamp(min=1.0)
        nu = mask_b.float() / mask_b.sum(-1, keepdim=True).clamp(min=1.0)

        # Run Sinkhorn
        plan, u, v = unbalanced_sinkhorn(
            c=cost_matrix,
            a=mu,
            b=nu,
            num_iter=num_iter,
            reg=reg,
            lambda1=lambda1,
            lambda2=lambda2,
            mask_a=mask_a,
            mask_b=mask_b,
            u_init=u_init,
            v_init=v_init,
            **kwargs,
        )

        return {
            "transport_plan": plan,
            "scaling_u": u,
            "scaling_v": v,
            "cost_matrix": cost_matrix,
            "mu": mu,
            "nu": nu,
        }
