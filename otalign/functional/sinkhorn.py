import torch


class Sinkhorn(torch.autograd.Function):
    """Sinkhorn iteration with entropy regularization and optional masks.

    This class implements the Sinkhorn algorithm as a PyTorch autograd function,
    allowing for automatic differentiation. It supports entropy regularization
    and optional boolean masks to deactivate specific entries in the marginals.

    Args:
        ctx: PyTorch context object to save tensors for backward pass.
        c: Cost matrix, shape [..., m, n].
        a: Source marginals, shape [..., m].
        b: Target marginals, shape [..., n].
        num_iter: Number of Sinkhorn iterations (positive integer).
        reg: Regularization parameter (positive float).
        mask_a: Boolean mask for source marginals, shape [..., m] or None.
        mask_b: Boolean mask for target marginals, shape [..., n] or None.

    Returns:
        Transport plan p, shape [..., m, n].

    Notes:
        - If mask_a or mask_b is None, all entries are considered active.
        - Masked positions (False) in a and b are set to zero and excluded.
        - Cost matrix c is adjusted to a large value at masked positions.
    """

    @staticmethod
    def forward(ctx, c, a, b, num_iter, reg, mask_a=None, mask_b=None):
        # Prepare masks: default to all True if None.
        if mask_a is None:
            mask_a = torch.ones_like(a, dtype=torch.bool)
        if mask_b is None:
            mask_b = torch.ones_like(b, dtype=torch.bool)

        # Broadcast masks to match cost matrix dimensions.
        row_mask = mask_a.unsqueeze(-1)  # [..., m, 1]
        col_mask = mask_b.unsqueeze(-2)  # [..., 1, n]
        full_mask = row_mask & col_mask  # [..., m, n]

        # Zero out forbidden marginals and normalize.
        a = a * mask_a
        b = b * mask_b
        eps = 1e-12  # Small value to prevent division by zero.
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=eps)
        b = b / b.sum(dim=-1, keepdim=True).clamp(min=eps)

        # Set high cost for forbidden entries.
        big_cost = 1e6
        c = c.masked_fill(~row_mask, big_cost)
        c = c.masked_fill(~col_mask, big_cost)

        # Sinkhorn iterations in log-space for stability.
        log_p = -c / reg
        log_a = torch.log(a + eps).unsqueeze(-1)
        log_b = torch.log(b + eps).unsqueeze(-2)
        for _ in range(num_iter):
            log_p -= torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b
            log_p -= torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a
        p = torch.exp(log_p) * full_mask  # Masked positions become 0.

        # Save tensors for backward pass.
        ctx.save_for_backward(
            p,
            p.sum(dim=-1).clamp(min=eps),  # Row sums.
            p.sum(dim=-2).clamp(min=eps),  # Column sums.
            mask_a,
            mask_b,
        )
        ctx.reg = reg
        return p

    @staticmethod
    def backward(ctx, grad_p):
        # Unpack saved tensors and parameters.
        p, a_sum, b_sum, mask_a, mask_b = ctx.saved_tensors
        reg = ctx.reg
        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        # Zero gradients at masked positions.
        full_mask = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        grad_p = grad_p.masked_fill(~full_mask, 0)

        # Implicit differentiation for gradients.
        grad_p = -p * grad_p / reg
        k = torch.cat(
            (
                torch.cat((torch.diag_embed(a_sum), p), dim=-1),
                torch.cat((p.transpose(-2, -1), torch.diag_embed(b_sum)), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]
        t = torch.cat((grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1).unsqueeze(-1)

        # Solve linear system with damping for stability.
        damp = 1e-9
        eye = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
        grad_ab = torch.linalg.solve(k + damp * eye, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat(
            (grad_ab[..., m:, :], grad_ab.new_zeros(batch_shape + [1, 1])),
            dim=-2,
        )

        # Compute gradient w.r.t. cost matrix.
        u = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * u

        # Apply masks and scale gradients.
        grad_a = (-reg * grad_a.squeeze(-1)) * mask_a
        grad_b = (-reg * grad_b.squeeze(-1)) * mask_b

        # Return gradients for all forward arguments.
        return grad_p, grad_a, grad_b, None, None, None, None


# Convenience wrapper for direct use.
_sinkhorn = Sinkhorn.apply


def sinkhorn(c, a, b, num_iter, reg, mask_a=None, mask_b=None) -> torch.Tensor:
    """
    A wrapper for the Sinkhorn algorithm that performs input validation to handle edge cases.

    This function checks the types, shapes, and values of the inputs to ensure they meet the
    requirements of the Sinkhorn algorithm. It raises informative errors if any assumptions are
    violated, preventing unexpected behavior or runtime errors.

    Args:
        c (torch.Tensor): Cost matrix, shape [..., m, n].
        a (torch.Tensor): Source marginals, shape [..., m].
        b (torch.Tensor): Target marginals, shape [..., n].
        num_iter (int): Number of Sinkhorn iterations (must be a positive integer).
        reg (float): Regularization parameter (must be a positive scalar).
        mask_a (torch.BoolTensor or None): Mask for source marginals, shape [..., m] or None.
        mask_b (torch.BoolTensor or None): Mask for target marginals, shape [..., n] or None.

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

    # Call the original sinkhorn function
    return _sinkhorn(c, a, b, num_iter, reg, mask_a, mask_b)
