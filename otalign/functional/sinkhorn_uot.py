import warnings

import torch


class SinkhornUOT(torch.autograd.Function):
    """Sinkhorn iteration for Unbalanced Optimal Transport with entropy regularization and optional masks.

    This class implements the Sinkhorn algorithm for UOT with improved numerical stability.
    It supports entropy regularization, KL divergence penalties for marginal relaxation,
    and optional boolean masks to deactivate specific entries.

    Args:
        ctx: PyTorch context object to save tensors for backward pass.
        c: Cost matrix, shape [..., m, n].
        a: Source marginals, shape [..., m].
        b: Target marginals, shape [..., n].
        num_iter: Number of Sinkhorn iterations (positive integer).
        reg: Entropy regularization parameter (positive float).
        lambda1: KL divergence penalty for source marginals (positive float).
        lambda2: KL divergence penalty for target marginals (positive float).
        mask_a: Boolean mask for source marginals, shape [..., m] or None.
        mask_b: Boolean mask for target marginals, shape [..., n] or None.
        damp: Damping factor for linear system (default: 1e-6).
        min_marginal: Minimum value for marginal sums to avoid division by zero (default: 1e-8).

    Returns:
        Transport plan p, shape [..., m, n].

    """

    @staticmethod
    def forward(
        ctx,
        c,
        a,
        b,
        num_iter,
        reg,
        lambda1,
        lambda2,
        mask_a=None,
        mask_b=None,
        damp=1e-6,
        min_marginal=1e-8,
        tol=1e-4,
    ):
        # Prepare masks: default to all True if None.
        if mask_a is None:
            mask_a = torch.ones_like(a, dtype=torch.bool)
        if mask_b is None:
            mask_b = torch.ones_like(b, dtype=torch.bool)

        # Broadcast masks to match cost matrix dimensions.
        row_mask = mask_a.unsqueeze(-1)  # [..., m, 1]
        col_mask = mask_b.unsqueeze(-2)  # [..., 1, n]
        full_mask = row_mask & col_mask  # [..., m, n]

        # Zero out forbidden marginals.
        a = a * mask_a
        b = b * mask_b
        eps = 1e-8  # Increased for stability.

        # Set high cost for forbidden entries.
        big_cost = 1e6
        c = c.masked_fill(~row_mask, big_cost)
        c = c.masked_fill(~col_mask, big_cost)

        # Sinkhorn iterations in log-space for stability.
        log_p = -c / reg
        log_a = torch.log(a + eps).unsqueeze(-1)
        log_b = torch.log(b + eps).unsqueeze(-2)
        alpha = torch.zeros_like(log_a)
        beta = torch.zeros_like(log_b)

        # Scaling factors for UOT
        scale_a = lambda1 / (lambda1 + reg)
        scale_b = lambda2 / (lambda2 + reg)

        # Track convergence
        converged = False
        for _ in range(num_iter):
            prev_p = torch.exp(log_p) * full_mask
            alpha = scale_a * (log_a - torch.logsumexp(log_p + beta, dim=-2, keepdim=True))
            beta = scale_b * (log_b - torch.logsumexp(log_p + alpha, dim=-1, keepdim=True))
            log_p = -c / reg + alpha + beta
            p = torch.exp(log_p) * full_mask

            # Convergence check: relative change in pi
            diff_p = torch.norm(p - prev_p, p=1) / (torch.norm(prev_p, p=1) + eps)
            if diff_p < tol:
                converged = True
                break

        # Warn if maximum iterations reached without convergence
        if not converged:
            warnings.warn(
                f"Sinkhorn did not converge after {num_iter} iterations; final pi change = {diff_p.item():.6f}",
            )

        # Clip marginal sums to avoid division by zero
        p_sum_row = p.sum(dim=-1).clamp(min=min_marginal)
        p_sum_col = p.sum(dim=-2).clamp(min=min_marginal)

        # Save tensors for backward pass.
        ctx.save_for_backward(p, p_sum_row, p_sum_col, mask_a, mask_b)
        ctx.reg = reg
        ctx.lambda1 = lambda1
        ctx.lambda2 = lambda2
        ctx.damp = damp
        return p

    @staticmethod
    def backward(ctx, grad_p):
        # Unpack saved tensors and parameters.
        p, a_sum, b_sum, mask_a, mask_b = ctx.saved_tensors
        reg, lambda1, lambda2, damp = ctx.reg, ctx.lambda1, ctx.lambda2, ctx.damp
        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        # Zero gradients at masked positions.
        full_mask = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        grad_p = grad_p.masked_fill(~full_mask, 0)

        # Implicit differentiation for gradients.
        grad_p_scaled = -p * grad_p / reg
        k = torch.cat(
            (
                torch.cat((torch.diag_embed(a_sum / (lambda1 + reg)), p / (lambda1 + reg)), dim=-1),
                torch.cat((p.transpose(-2, -1) / (lambda2 + reg), torch.diag_embed(b_sum / (lambda2 + reg))), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]
        t = torch.cat((grad_p_scaled.sum(dim=-1), grad_p_scaled[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)

        # Solve linear system with damping; fallback to pseudo-inverse if singular.
        eye = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
        try:
            grad_ab = torch.linalg.solve(k + damp * eye, t)
        except RuntimeError as e:
            warnings.warn(f"Singular matrix detected: {e!s}. Using pseudo-inverse.")
            grad_ab = torch.linalg.pinv(k + damp * eye) @ t

        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat(
            (grad_ab[..., m:, :], grad_ab.new_zeros(batch_shape + [1, 1])),
            dim=-2,
        )

        # Compute gradient w.r.t. cost matrix.
        u = grad_a + grad_b.transpose(-2, -1)
        grad_p_scaled -= p * u

        # Apply masks and scale gradients.
        grad_a = (-reg * grad_a.squeeze(-1)) * mask_a
        grad_b = (-reg * grad_b.squeeze(-1)) * mask_b

        # Return gradients for all forward arguments.
        return grad_p_scaled, grad_a, grad_b, None, None, None, None, None, None, None, None, None


# Convenience wrapper for direct use.
_sinkhorn_uot = SinkhornUOT.apply


def unbalanced_sinkhorn(
    c,
    a,
    b,
    num_iter,
    reg,
    lambda1,
    lambda2,
    mask_a=None,
    mask_b=None,
    damp=1e-6,
    min_marginal=1e-8,
    tol=1e-4,
) -> torch.Tensor:
    """A wrapper for the Unbalanced Sinkhorn algorithm with improved stability.

    Args:
        c (torch.Tensor): Cost matrix, shape [..., m, n].
        a (torch.Tensor): Source marginals, shape [..., m].
        b (torch.Tensor): Target marginals, shape [..., n].
        num_iter (int): Number of Sinkhorn iterations (must be a positive integer).
        reg (float): Entropy regularization parameter (must be a positive scalar).
        lambda1 (float): KL divergence penalty for source marginals (positive scalar).
        lambda2 (float): KL divergence penalty for target marginals (positive scalar).
        mask_a (torch.BoolTensor or None): Mask for source marginals, shape [..., m] or None.
        mask_b (torch.BoolTensor or None): Mask for target marginals, shape [..., n] or None.
        damp (float): Damping factor for linear system (default: 1e-6).
        min_marginal (float): Minimum value for marginal sums (default: 1e-8).
        tol (float): Convergence tolerance for relative change in pi (default: 1e-4).

    Returns:
        torch.Tensor: The computed transport plan, shape [..., m, n].

    Raises:
        TypeError: If inputs are not of the expected types.
        ValueError: If inputs have incorrect shapes or invalid values.

    """
    # Check types
    if not torch.is_tensor(c):
        raise TypeError("c must be a torch.Tensor")
    if not torch.is_tensor(a):
        raise TypeError("a must be a torch.Tensor")
    if not torch.is_tensor(b):
        raise TypeError("b must be a torch.Tensor")
    if mask_a is not None:
        if not torch.is_tensor(mask_a):
            raise TypeError("mask_a must be a torch.Tensor")
        if mask_a.dtype != torch.bool:
            raise TypeError("mask_a must be a boolean tensor")
    if mask_b is not None:
        if not torch.is_tensor(mask_b):
            raise TypeError("mask_b must be a torch.Tensor")
        if mask_b.dtype != torch.bool:
            raise TypeError("mask_b must be a boolean tensor")
    if not isinstance(num_iter, int) or num_iter <= 0:
        raise ValueError("num_iter must be a positive integer")
    if not isinstance(reg, (int, float)) or reg <= 0:
        raise ValueError("reg must be a positive scalar")
    if not isinstance(lambda1, (int, float)) or lambda1 <= 0:
        raise ValueError("lambda1 must be a positive scalar")
    if not isinstance(lambda2, (int, float)) or lambda2 <= 0:
        raise ValueError("lambda2 must be a positive scalar")
    if not isinstance(damp, (int, float)) or damp <= 0:
        raise ValueError("damp must be a positive scalar")
    if not isinstance(min_marginal, (int, float)) or min_marginal <= 0:
        raise ValueError("min_marginal must be a positive scalar")
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive scalar")

    # Check shapes
    batch_shape = c.shape[:-2]
    m, n = c.shape[-2:]
    if m <= 0 or n <= 0:
        raise ValueError("Dimensions m and n must be positive")
    if a.shape != batch_shape + (m,):
        raise ValueError(f"a has incorrect shape: expected {batch_shape + (m,)}, got {a.shape}")
    if b.shape != batch_shape + (n,):
        raise ValueError(f"b has incorrect shape: expected {batch_shape + (n,)}, got {b.shape}")
    if mask_a is not None and mask_a.shape != batch_shape + (m,):
        raise ValueError(f"mask_a has incorrect shape: expected {batch_shape + (m,)}, got {mask_a.shape}")
    if mask_b is not None and mask_b.shape != batch_shape + (n,):
        raise ValueError(f"mask_b has incorrect shape: expected {batch_shape + (n,)}, got {mask_b.shape}")

    # Check values
    if not (a >= 0).all():
        raise ValueError("a must be non-negative")
    if not (b >= 0).all():
        raise ValueError("b must be non-negative")
    if not torch.isfinite(c).all():
        warnings.warn("Non-finite values detected in cost matrix c")

    # Call the unbalanced sinkhorn function
    return _sinkhorn_uot(c, a, b, num_iter, reg, lambda1, lambda2, mask_a, mask_b, damp, min_marginal, tol)  # type: ignore
