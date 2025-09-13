import warnings

import torch


@torch.no_grad()
def safe_delta(f_new, f_old, mask):
    valid = mask & torch.isfinite(f_new) & torch.isfinite(f_old)
    if valid.any():
        return (f_new[valid] - f_old[valid]).abs().max()
    else:
        return torch.tensor(float("nan"), device=f_new.device, dtype=f_new.dtype)


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

    Returns:
        A tuple containing:
        - p (torch.Tensor): Transport plan, shape [..., m, n].
        - u (torch.Tensor): Source scaling vector, shape [..., m].
        - v (torch.Tensor): Target scaling vector, shape [..., n].

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
        u_init=None,
        v_init=None,
        eps=1e-12,
        damp=1e-6,
        tol=1e-4,
    ):
        if mask_a is None:
            mask_a = torch.ones_like(a, dtype=torch.bool)
        if mask_b is None:
            mask_b = torch.ones_like(b, dtype=torch.bool)

        # Mask cost with +inf → log-kernel has -inf at forbidden entries.
        inf = torch.tensor(float("inf"), device=c.device, dtype=c.dtype)
        c_masked = c.masked_fill(~mask_a.unsqueeze(-1), inf)
        c_masked = c_masked.masked_fill(~mask_b.unsqueeze(-2), inf)
        logK = -c_masked / reg

        # Log-marginals; masked positions force -inf so u,v -> 0 there.
        ninf_m = torch.full_like(a, -float("inf"))
        ninf_n = torch.full_like(b, -float("inf"))
        log_a = torch.where(mask_a, torch.log(a.clamp_min(eps)), ninf_m)
        log_b = torch.where(mask_b, torch.log(b.clamp_min(eps)), ninf_n)

        # Initialize duals (log u, log v)
        if u_init is None:
            f = torch.zeros_like(a)  # [..., m]
        else:
            f = torch.log(u_init)
        if v_init is None:
            g = torch.zeros_like(b)  # [..., n]
        else:
            g = torch.log(v_init)

        tau_a = lambda1 / (lambda1 + reg)
        tau_b = lambda2 / (lambda2 + reg)

        converged = False
        for _ in range(1, max(int(num_iter + 1), 1)):
            f_prev, g_prev = f.clone(), g.clone()

            # f update: log a - logsumexp(logK + g)
            s = torch.logsumexp(logK + g.unsqueeze(-2), dim=-1)  # [B,M]
            f = tau_a * (log_a - s)
            f = torch.where(mask_a, f, ninf_m)

            # g update: log b - logsumexp(logK^T + f)
            t = torch.logsumexp(logK + f.unsqueeze(-1), dim=-2)  # [B,N]
            g = tau_b * (log_b - t)
            g = torch.where(mask_b, g, ninf_n)

            # Convergence on duals (small vectors, not the full plan)
            df = safe_delta(f, f_prev, mask_a)
            dg = safe_delta(g, g_prev, mask_b)
            if torch.max(df, dg) < tol:
                converged = True
                break

        if not converged and num_iter > 0:
            msg = f"Sinkhorn (UOT) did not reach tol={tol} after {num_iter} iters; last df={df:.5e}, dg={dg:.5e}"
            warnings.warn(msg)

        # Build plan once at the end (masked entries become 0 since -inf)
        joint = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        logP = logK + f.unsqueeze(-1) + g.unsqueeze(-2)
        logP = torch.where(joint, logP, torch.tensor(-float("inf"), device=logP.device, dtype=logP.dtype))

        P = torch.exp(logP)  # [..., m, n]
        u = torch.exp(f)  # [..., m]
        v = torch.exp(g)  # [..., n]

        # Save for backward
        ctx.save_for_backward(P, u, v, a, b, mask_a, mask_b)
        ctx.reg = reg
        ctx.lambda1 = lambda1
        ctx.lambda2 = lambda2
        ctx.damp = damp

        return P, u, v

    @staticmethod
    def backward(ctx, grad_p, grad_u, grad_v):
        # Unpack saved tensors and parameters.
        p, u, v, a, b, mask_a, mask_b = ctx.saved_tensors
        reg, lambda1, lambda2, damp = ctx.reg, ctx.lambda1, ctx.lambda2, ctx.damp

        # Handle None gradients
        grad_p = torch.zeros_like(p) if grad_p is None else grad_p
        grad_u = torch.zeros_like(u) if grad_u is None else grad_u
        grad_v = torch.zeros_like(v) if grad_v is None else grad_v

        # Zero gradients at masked positions.
        full_mask = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        grad_p = grad_p.masked_fill(~full_mask, 0)

        # Calculate the total gradient flowing into the fixed-point variables (alpha, beta)
        # from all three outputs (p, u, v).
        # dL/d_alpha = (dL/dp)*(dp/d_alpha) + (dL/du)*(du/d_alpha) = (grad_p * p).sum(-1) + grad_u * u
        # dL/d_beta = (dL/dp)*(dp/d_beta) + (dL/dv)*(dv/d_beta) = (grad_p * p).sum(-2) + grad_v * v
        grad_alpha_total = (grad_p * p).sum(dim=-1) + grad_u * u
        grad_beta_total = (grad_p * p).sum(dim=-2) + grad_v * v

        # This vector `t` is the right-hand side of the linear system in the adjoint method.
        # It represents the total incoming gradient that needs to be propagated backwards.
        # The slicing [..., :-1] is to make the system non-singular.
        t = torch.cat((grad_alpha_total, grad_beta_total[..., :-1]), dim=-1).unsqueeze(-1)

        # Recompute marginals of the final plan p for the Jacobian matrix k
        a_sum = p.sum(dim=-1).clamp(min=1e-8)
        b_sum = p.sum(dim=-2).clamp(min=1e-8)

        # Construct the Jacobian matrix `k` for the linear system.
        # This matrix represents the derivative of the fixed-point operator.
        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        k = torch.cat(
            (
                torch.cat((torch.diag_embed(a_sum / (lambda1 + reg)), p / (lambda1 + reg)), dim=-1),
                torch.cat((p.transpose(-2, -1) / (lambda2 + reg), torch.diag_embed(b_sum / (lambda2 + reg))), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]

        # Solve the linear system to get the adjoints (z_alpha, z_beta).
        eye = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
        try:
            z = torch.linalg.solve(k + damp * eye, t)
        except RuntimeError as e:
            warnings.warn(f"Singular matrix detected: {e!s}. Using pseudo-inverse.")
            z = torch.linalg.pinv(k + damp * eye) @ t

        # Unpack the adjoints
        z_alpha = z[..., :m, :]
        z_beta = torch.cat(
            (z[..., m:, :], z.new_zeros(batch_shape + [1, 1])),
            dim=-2,
        )

        # Compute gradients w.r.t. inputs a and b using the adjoints.
        scale_a = lambda1 / (lambda1 + reg)
        scale_b = lambda2 / (lambda2 + reg)

        # dL/da = (dL/d_alpha) * (d_alpha/da) = z_alpha * scale_a / a
        grad_a = (z_alpha.squeeze(-1) * scale_a / (a + 1e-8)) * mask_a
        grad_b = (z_beta.squeeze(-1) * scale_b / (b + 1e-8)) * mask_b

        # Compute gradient w.r.t. cost matrix c.
        # It has a direct component and an indirect component (from the adjoints).
        direct_grad_c = grad_p * (-p / reg)
        indirect_grad_c = -p * (z_alpha + z_beta.transpose(-2, -1)) / reg
        grad_c = direct_grad_c + indirect_grad_c
        grad_c = grad_c.masked_fill(~full_mask, 0)

        # Return gradients for all forward arguments.
        return grad_c, grad_a, grad_b, None, None, None, None, None, None, None, None, None, None, None


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
    u_init=None,
    v_init=None,
    damp=1e-6,
    tol=1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        tol (float): Convergence tolerance for relative change in pi (default: 1e-4).

    Returns:
        A tuple containing:
        - p (torch.Tensor): The computed transport plan, shape [..., m, n].
        - u (torch.Tensor): The final source scaling vector, shape [..., m].
        - v (torch.Tensor): The final target scaling vector, shape [..., n].

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
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive scalar")
    if u_init is not None:
        if not torch.is_tensor(u_init):
            raise TypeError("u_init must be a torch.Tensor")
    if v_init is not None:
        if not torch.is_tensor(v_init):
            raise TypeError("v_init must be a torch.Tensor")

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
    if u_init is not None and u_init.shape != batch_shape + (m,):
        raise ValueError(f"u_init has incorrect shape: expected {batch_shape + (m,)}, got {u_init.shape}")
    if v_init is not None and v_init.shape != batch_shape + (n,):
        raise ValueError(f"v_init has incorrect shape: expected {batch_shape + (n,)}, got {v_init.shape}")

    # Check values
    if not (a >= 0).all():
        raise ValueError("a must be non-negative")
    if not (b >= 0).all():
        raise ValueError("b must be non-negative")
    if not torch.isfinite(c).all():
        warnings.warn("Non-finite values detected in cost matrix c")

    # Call the unbalanced sinkhorn function
    return _sinkhorn_uot(c, a, b, num_iter, reg, lambda1, lambda2, mask_a, mask_b, u_init, v_init, damp, tol)  # type: ignore
