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
    """
    Unbalanced Sinkhorn with optional masks, optional POT-like stabilization (absorption),
    and configurable gauge fixing for the implicit backward (adjoint solve).
    """

    @staticmethod
    def forward(
        ctx,
        c: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        num_iter: int,
        reg: float,
        lambda1: float,
        lambda2: float,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
        u_init: torch.Tensor | None = None,
        v_init: torch.Tensor | None = None,
        eps: float = 1e-12,
        damp: float = 1e-6,
        tol: float = 1e-4,
        gauge: str = "last_beta",  # "last_beta" or "mean_zero_beta"
        stabilize: bool = False,  # POT-like absorption
        tau: float = 1e5,  # absorption threshold
        adjoint_solver: str = "dense",  # "dense" | "schur_cg"
        cg_tol: float = 1e-12,
        cg_max_iter: int = 2000,
    ):
        if mask_a is None:
            mask_a = torch.ones_like(a, dtype=torch.bool)
        if mask_b is None:
            mask_b = torch.ones_like(b, dtype=torch.bool)

        # Mask costs to +inf (forbidden pairs)
        inf = torch.tensor(float("inf"), device=c.device, dtype=c.dtype)
        c_masked = c.masked_fill(~mask_a.unsqueeze(-1), inf)
        c_masked = c_masked.masked_fill(~mask_b.unsqueeze(-2), inf)

        # Log-kernel
        logK = -c_masked / reg

        # Log marginals (masked -> -inf)
        ninf_m = torch.full_like(a, -float("inf"))
        ninf_n = torch.full_like(b, -float("inf"))
        log_a = torch.where(mask_a, torch.log(a.clamp_min(eps)), ninf_m)
        log_b = torch.where(mask_b, torch.log(b.clamp_min(eps)), ninf_n)

        # Dual logs f=log u, g=log v
        f = torch.zeros_like(a) if u_init is None else torch.log(u_init.clamp_min(eps))
        g = torch.zeros_like(b) if v_init is None else torch.log(v_init.clamp_min(eps))

        # Offsets for POT-like stabilization
        alpha_off = torch.zeros_like(a)
        beta_off = torch.zeros_like(b)

        # UOT weights
        tau_a = lambda1 / (lambda1 + reg)
        tau_b = lambda2 / (lambda2 + reg)

        converged = False
        for i in range(1, max(int(num_iter + 1), 1)):
            f_prev, g_prev = f.clone(), g.clone()

            # f update
            s = torch.logsumexp(logK + g.unsqueeze(-2), dim=-1)  # log(Kv)
            f = tau_a * (log_a - s)
            f = torch.where(mask_a, f, ninf_m)

            # g update
            t = torch.logsumexp(logK + f.unsqueeze(-1), dim=-2)  # log(K^T u)
            g = tau_b * (log_b - t)
            g = torch.where(mask_b, g, ninf_n)

            # Optional stabilization (absorption)
            if stabilize:
                u_tmp = torch.where(mask_a, torch.exp(f), torch.zeros_like(f))
                v_tmp = torch.where(mask_b, torch.exp(g), torch.zeros_like(g))

                need_absorb_u = (u_tmp > tau).any(dim=-1, keepdim=True)
                need_absorb_v = (v_tmp > tau).any(dim=-1, keepdim=True)

                if bool(need_absorb_u.any() or need_absorb_v.any()):
                    max_u = torch.where(
                        need_absorb_u,
                        u_tmp.amax(dim=-1, keepdim=True).clamp_min(1.0),
                        torch.ones_like(u_tmp[..., :1]),
                    )
                    max_v = torch.where(
                        need_absorb_v,
                        v_tmp.amax(dim=-1, keepdim=True).clamp_min(1.0),
                        torch.ones_like(v_tmp[..., :1]),
                    )
                    a_shift = torch.zeros_like(a)
                    b_shift = torch.zeros_like(b)
                    a_shift = torch.where(need_absorb_u.squeeze(-1), (reg * max_u.log()).squeeze(-1), a_shift)
                    b_shift = torch.where(need_absorb_v.squeeze(-1), (reg * max_v.log()).squeeze(-1), b_shift)
                    alpha_off = alpha_off + a_shift
                    beta_off = beta_off + b_shift

                    # Rebuild logK with offsets
                    logK = (alpha_off.unsqueeze(-1) + beta_off.unsqueeze(-2) - c_masked) / reg
                    # Reset g (as in POT)
                    g = torch.zeros_like(g)

            # Periodic convergence check
            if i % 10 == 0:
                df = safe_delta(f, f_prev, mask_a)
                dg = safe_delta(g, g_prev, mask_b)
                if torch.max(df, dg) < tol:
                    converged = True
                    break

        if not converged and num_iter > 0:
            warnings.warn(f"Sinkhorn (UOT) did not reach tol={tol} after {num_iter} iters.")

        joint = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        # Final log plan with offsets
        logP = (alpha_off.unsqueeze(-1) + beta_off.unsqueeze(-2) - c_masked) / reg + f.unsqueeze(-1) + g.unsqueeze(-2)
        logP = torch.where(joint, logP, torch.tensor(-float("inf"), device=logP.device, dtype=logP.dtype))

        P = torch.exp(logP)
        u = torch.exp(f)
        v = torch.exp(g)

        # Save for backward
        ctx.save_for_backward(P, u, v, a, b, mask_a, mask_b)
        ctx.reg = reg
        ctx.lambda1 = lambda1
        ctx.lambda2 = lambda2
        ctx.damp = damp
        ctx.eps = eps
        ctx.gauge = gauge
        ctx.adjoint_solver = adjoint_solver
        ctx.cg_tol = cg_tol
        ctx.cg_max_iter = cg_max_iter

        return P, u, v

    @staticmethod
    def backward(ctx, grad_p, grad_u, grad_v):
        p, u, v, a, b, mask_a, mask_b = ctx.saved_tensors
        reg, lambda1, lambda2, damp = ctx.reg, ctx.lambda1, ctx.lambda2, ctx.damp
        eps, gauge = ctx.eps, ctx.gauge
        adjoint_solver, cg_tol, cg_max_iter = ctx.adjoint_solver, ctx.cg_tol, ctx.cg_max_iter

        grad_p = torch.zeros_like(p) if grad_p is None else grad_p
        grad_u = torch.zeros_like(u) if grad_u is None else grad_u
        grad_v = torch.zeros_like(v) if grad_v is None else grad_v

        full_mask = mask_a.unsqueeze(-1) & mask_b.unsqueeze(-2)
        grad_p = grad_p.masked_fill(~full_mask, 0)

        # t = [t_phi; t_psi]
        G = grad_p
        t_phi = (G * p).sum(dim=-1) + grad_u * u
        t_psi = (G * p).sum(dim=-2) + grad_v * v

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        tau_a = lambda1 / (lambda1 + reg)
        tau_b = lambda2 / (lambda2 + reg)

        p_marg = p.sum(dim=-1).clamp(min=eps)  # [..., m]
        q_marg = p.sum(dim=-2).clamp(min=eps)  # [..., n]

        # Diagonals
        Dp_inv = torch.diag_embed(1.0 / p_marg)  # = diag(p)^{-1}
        Dq_inv = torch.diag_embed(1.0 / q_marg)  # = diag(q)^{-1}

        # Build J and JT only if using dense solver
        def solve_adjoint_dense(t_phi, t_psi):
            eye_m = torch.eye(m, device=p.device, dtype=p.dtype).expand(batch_shape + [m, m])
            eye_n = torch.eye(n, device=p.device, dtype=p.dtype).expand(batch_shape + [n, n])

            J_12 = tau_a * (Dp_inv @ p)  # [..., m, n]
            J_21 = tau_b * (Dq_inv @ p.transpose(-2, -1))  # [..., n, m]

            J_top = torch.cat((eye_m, J_12), dim=-1)
            J_bot = torch.cat((J_21, eye_n), dim=-1)
            J = torch.cat((J_top, J_bot), dim=-2)  # [..., m+n, m+n]

            JT = J.transpose(-2, -1)
            t_full = torch.cat((t_phi, t_psi), dim=-1).unsqueeze(-1)

            if gauge == "last_beta":
                K = JT[..., :-1, :-1]
                rhs = t_full[..., :-1, :]
                Id = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
                try:
                    lam = torch.linalg.solve(K + damp * Id, rhs)
                except RuntimeError as e:
                    warnings.warn(f"Singular system (adjoint dense): {e!s}. Using pseudo-inverse.")
                    lam = torch.linalg.pinv(K + damp * Id) @ rhs
                lam = torch.cat((lam, lam.new_zeros(batch_shape + [1, 1])), dim=-2)

            elif gauge == "mean_zero_beta":
                c_vec = torch.zeros(batch_shape + [m + n, 1], device=JT.device, dtype=JT.dtype)
                c_vec[..., m:, 0] = 1.0
                zero11 = torch.zeros(batch_shape + [1, 1], device=JT.device, dtype=JT.dtype)
                K11 = JT + damp * torch.eye(m + n, device=JT.device, dtype=JT.dtype).expand_as(JT)
                K12 = c_vec
                K21 = c_vec.transpose(-2, -1)
                K22 = zero11
                K_top = torch.cat((K11, K12), dim=-1)
                K_bot = torch.cat((K21, K22), dim=-1)
                KKT = torch.cat((K_top, K_bot), dim=-2)
                rhs = torch.cat((t_full, torch.zeros_like(zero11)), dim=-2)
                try:
                    sol = torch.linalg.solve(KKT, rhs)
                except RuntimeError as e:
                    warnings.warn(f"Singular system (adjoint KKT): {e!s}. Using pseudo-inverse.")
                    sol = torch.linalg.pinv(KKT) @ rhs
                lam = sol[..., : (m + n), :]

            else:
                raise ValueError(f"Unknown gauge: {gauge}")

            lam_phi = lam[..., :m, 0]
            lam_psi = lam[..., m:, 0]
            return lam_phi, lam_psi

        def solve_adjoint_schur_cg(t_phi, t_psi):
            # Schur complement on lambda_psi using JT structure:
            # lambda_phi = t_phi - tau_b * P * Dq^{-1} * lambda_psi
            # (I - tau_a*tau_b * P^T Dp^{-1} P Dq^{-1}) lambda_psi = t_psi - tau_a * P^T Dp^{-1} t_phi

            # Precompute b_psi and define matvec for symmetric reweighted system
            b_psi = t_psi - tau_a * (p.transpose(-2, -1) @ (Dp_inv @ t_phi.unsqueeze(-1))).squeeze(-1)  # [..., n]

            # Work with x = Dq^{-1/2} lambda_psi for symmetry: S' = I - tau_a*tau_b * A^T A
            Dq_mhalf = (1.0 / q_marg.sqrt()).unsqueeze(-1)  # [..., n, 1]
            Dq_half = q_marg.sqrt().unsqueeze(-1)  # [..., n, 1]
            Dp_mhalf = (1.0 / p_marg.sqrt()).unsqueeze(-1)  # [..., m, 1]

            # b' = Dq^{-1/2} b_psi
            b_prime = (b_psi.unsqueeze(-1) * Dq_mhalf).squeeze(-1)

            def A_times(x):  # x: [..., n]
                # y = A x = Dp^{-1/2} P Dq^{-1/2} x
                y = (x.unsqueeze(-1) * Dq_mhalf).squeeze(-1)  # scale by Dq^{-1/2}
                y = p @ y.unsqueeze(-1)  # P * (...)
                y = y.squeeze(-1) * Dp_mhalf.squeeze(-1)  # Dp^{-1/2} *
                return y  # [..., m]

            def AT_times(y):  # y: [..., m]
                # z = A^T y = Dq^{-1/2} P^T Dp^{-1/2} y
                z = (y.unsqueeze(-1) * Dp_mhalf).squeeze(-1)  # Dp^{-1/2} *
                z = p.transpose(-2, -1) @ z.unsqueeze(-1)  # P^T *
                z = z.squeeze(-1) * Dq_mhalf.squeeze(-1)  # Dq^{-1/2} *
                return z  # [..., n]

            def S_prime(x):  # (I - tau_a*tau_b * A^T A + damp*I) x
                Ax = A_times(x)
                ATAx = AT_times(Ax)
                return x - (tau_a * tau_b) * ATAx + damp * x

            # Simple batched CG
            x = torch.zeros_like(b_prime)
            r = b_prime - S_prime(x)
            pdir = r.clone()
            rr_old = (r * r).sum(dim=-1, keepdim=True)

            for _ in range(cg_max_iter):
                Ap = S_prime(pdir)
                alpha = rr_old / ((pdir * Ap).sum(dim=-1, keepdim=True).clamp_min(1e-30))
                x = x + alpha * pdir
                r = r - alpha * Ap
                rr_new = (r * r).sum(dim=-1, keepdim=True)
                if (rr_new.sqrt() <= cg_tol).all():
                    break
                beta = rr_new / rr_old
                pdir = r + beta * pdir
                rr_old = rr_new

            # Recover lambda_psi and lambda_phi
            lam_psi = (x.unsqueeze(-1) * Dq_half).squeeze(-1)  # lambda_psi = Dq^{1/2} x
            # lambda_phi = t_phi - tau_b * P * Dq^{-1} * lambda_psi
            lam_phi = t_phi - tau_b * (p @ ((lam_psi / q_marg).unsqueeze(-1))).squeeze(-1)
            return lam_phi, lam_psi

        if adjoint_solver == "schur_cg":
            lam_phi, lam_psi = solve_adjoint_schur_cg(t_phi, t_psi)
        else:
            lam_phi, lam_psi = solve_adjoint_dense(t_phi, t_psi)

        # Grads w.r.t. a, b
        grad_a = (tau_a * lam_phi / a.clamp_min(eps)) * mask_a
        grad_b = (tau_b * lam_psi / b.clamp_min(eps)) * mask_b

        # Corrections for C
        corr_alpha = (tau_a * lam_phi / p_marg).unsqueeze(-1)  # [..., m, 1]
        corr_beta = (tau_b * lam_psi / q_marg).unsqueeze(-2)  # [..., 1, n]

        grad_c = -(p / reg) * (G - (corr_alpha + corr_beta))
        grad_c = grad_c.masked_fill(~full_mask, 0)

        return (
            grad_c,
            grad_a,
            grad_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_sinkhorn_uot = SinkhornUOT.apply


def unbalanced_sinkhorn(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_iter: int,
    reg: float,
    lambda1: float,
    lambda2: float,
    mask_a: torch.Tensor | None = None,
    mask_b: torch.Tensor | None = None,
    u_init: torch.Tensor | None = None,
    v_init: torch.Tensor | None = None,
    eps: float = 1e-12,
    damp: float = 1e-6,
    tol: float = 1e-4,
    gauge: str = "last_beta",
    stabilize: bool = False,
    tau: float = 1e5,
    adjoint_solver: str = "dense",  # "dense" | "schur_cg"
    cg_tol: float = 1e-12,
    cg_max_iter: int = 2000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrapper for Unbalanced Sinkhorn with adjoint implicit backward, gauge fixing,
    and optional POT-like stabilization.
    """
    # Type checks
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
    if u_init is not None and not torch.is_tensor(u_init):
        raise TypeError("u_init must be a torch.Tensor")
    if v_init is not None and not torch.is_tensor(v_init):
        raise TypeError("v_init must be a torch.Tensor")
    if gauge not in ("last_beta", "mean_zero_beta"):
        raise ValueError("gauge must be 'last_beta' or 'mean_zero_beta'")

    # Shape checks
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

    # Value checks
    if not (a >= 0).all():
        raise ValueError("a must be non-negative")
    if not (b >= 0).all():
        raise ValueError("b must be non-negative")
    if not torch.isfinite(c).all():
        warnings.warn("Non-finite values detected in cost matrix c")

    # Call function; stash solver choices in ctx via attributes after call:
    P, u, v = _sinkhorn_uot(
        c,
        a,
        b,
        num_iter,
        reg,
        lambda1,
        lambda2,
        mask_a,
        mask_b,
        u_init,
        v_init,
        eps,
        damp,
        tol,
        gauge,
        stabilize,
        tau,
        adjoint_solver,
        cg_tol,
        cg_max_iter,
    )
    return P, u, v
