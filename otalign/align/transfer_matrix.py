# tm_gpu.py
# Transfer-matrix method for alignment on GPU with PyTorch + Triton.
# - Kernel A: anti-diagonal reductions -> LME[k], aD[k], bD[k], aI[k], bI[k]
# - Kernel B: persistent power-iteration over k for a batched set of mu -> pressure P_K(mu)
# - Convenience API: stats computation, pressure evaluation, mu_c bracketing/bisection, mu_peak via FD
#
# Requirements: torch>=2.1, triton>=2.1, CUDA GPU.

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def _neg_inf(dtype: torch.dtype) -> float:
    finfo = torch.finfo(dtype)
    return float(finfo.min / 4.0)


def _normalize_band(S: torch.Tensor, band: Optional[Tuple[int, int] | torch.Tensor]):
    """
    Normalize band:
      - None: full [0..n-1]
      - (jmin,jmax): static band
      - boolean mask (m,n): mask S outside band by -inf and return full j-range.
    """
    m, n = S.shape
    NEG = _neg_inf(S.dtype)
    if band is None:
        return 0, n - 1, S
    if isinstance(band, tuple) and len(band) == 2:
        return int(band[0]), int(band[1]), S
    if torch.is_tensor(band) and band.dtype == torch.bool and band.shape == (m, n):
        Sm = torch.where(band, S, torch.full_like(S, NEG))
        return 0, n - 1, Sm
    raise ValueError("band must be None, (jmin,jmax), or boolean mask of shape (m,n).")


# --------------------------- Kernel A: diagonal stats ---------------------------


@triton.jit
def diag_stats_kernel(
    # inputs
    S,  # (m, n) score matrix (log-scores before mu)
    gDo,
    gDe,  # (m,), (m,) deletion open/extend (row-wise)
    gIo,
    gIe,  # (n,), (n,) insertion open/extend (col-wise)
    m,
    n,  # ints
    gamma,  # float
    jmin,
    jmax,  # ints
    # outputs
    LME,
    aD,
    bD,
    aI,
    bI,  # (K,) each
    # meta
    K,  # m + n - 1
    BLOCK: tl.constexpr,  # number of (i) lanes processed per loop
):
    """
    Each program handles one anti-diagonal k (program_id(0)).
    It loops along t = 0..Lk-1 with a strided vector of BLOCK lanes,
    computing xmax (first pass), then sumexp, and linear means for gaps.
    """
    k = tl.program_id(0)

    # Bounds for this diagonal
    i_min = tl.maximum(0, k - (n - 1))
    i_max = tl.minimum(k, m - 1)
    # apply column band j in [jmin..jmax]
    i_min = tl.maximum(i_min, k - jmax)
    i_max = tl.minimum(i_max, k - jmin)
    Lk = i_max - i_min + 1

    if Lk <= 0:
        # Write sentinels for empty diagonal
        if k < K:
            tl.store(LME + k, -1e9)
            tl.store(aD + k, 0.0)
            tl.store(bD + k, 0.0)
            tl.store(aI + k, 0.0)
            tl.store(bI + k, 0.0)
        return

    # Pass 1: compute xmax along this diagonal for stable log-mean-exp
    # We iterate t in chunks of BLOCK (vector width).
    t0 = 0
    xmax = tl.full((), -1e30, tl.float32)
    while t0 < Lk:
        offs = t0 + tl.arange(0, BLOCK)
        mask = offs < Lk
        i = i_min + offs
        j = k - i
        offset = tl.where(mask, i * n + j, 0)
        sij = tl.load(S + offset, mask=mask, other=-1e9)  # Base is S (no b)
        x = gamma * sij
        x = tl.where(mask, x, -1e30)
        xmax = tl.maximum(xmax, tl.max(x, axis=0))
        t0 += BLOCK

    # Pass 2: compute sumexp and length + gap means
    t0 = 0
    sumexp = tl.zeros((), dtype=tl.float32)
    # Accumulate sums for gap means (row-wise for D, col-wise for I)
    sum_gDo = tl.zeros((), dtype=tl.float32)
    sum_gDe = tl.zeros((), dtype=tl.float32)
    sum_gIo = tl.zeros((), dtype=tl.float32)
    sum_gIe = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    while t0 < Lk:
        offs = t0 + tl.arange(0, BLOCK)
        mask = offs < Lk
        i = i_min + offs
        j = k - i
        # Fix: conditional offset for S
        offset = tl.where(mask, i * n + j, 0)
        sij = tl.load(S + offset, mask=mask, other=-1e9)
        x = gamma * sij
        expx = tl.exp(x - xmax)
        sumexp += tl.sum(tl.where(mask, expx, 0.0), axis=0)
        # Fix: conditional offsets for gaps
        i_offset = tl.where(mask, i, 0)
        j_offset = tl.where(mask, j, 0)
        gd_o = tl.load(gDo + i_offset, mask=mask, other=0.0)
        gd_e = tl.load(gDe + i_offset, mask=mask, other=0.0)
        gi_o = tl.load(gIo + j_offset, mask=mask, other=0.0)
        gi_e = tl.load(gIe + j_offset, mask=mask, other=0.0)
        sum_gDo += tl.sum(tl.where(mask, gd_o, 0.0), axis=0)
        sum_gDe += tl.sum(tl.where(mask, gd_e, 0.0), axis=0)
        sum_gIo += tl.sum(tl.where(mask, gi_o, 0.0), axis=0)
        sum_gIe += tl.sum(tl.where(mask, gi_e, 0.0), axis=0)
        count += tl.sum(tl.where(mask, 1.0, 0.0), axis=0)
        t0 += BLOCK

    # finalize
    lme = xmax + tl.log(sumexp / tl.maximum(count, 1.0))
    mean_gDo = sum_gDo / tl.maximum(count, 1.0)
    mean_gDe = sum_gDe / tl.maximum(count, 1.0)
    mean_gIo = sum_gIo / tl.maximum(count, 1.0)
    mean_gIe = sum_gIe / tl.maximum(count, 1.0)

    # store
    tl.store(LME + k, lme)
    tl.store(aD + k, mean_gDo)
    tl.store(bD + k, mean_gDe)
    tl.store(aI + k, mean_gIo)
    tl.store(bI + k, mean_gIe)


# --------------------------- Kernel B: pressure for batched mu ---------------------------


@triton.jit
def pressure_kernel(
    # stats over k
    LME,
    aD,
    bD,
    aI,
    bI,  # (K,)
    K,  # int
    gamma,  # float
    # mu-batch
    MU,  # (B,)
    B,  # int
    # outputs
    OUT_P,  # (B,) pressure
    # meta
    # no block lanes here: 1 program per mu (persistent loop over k)
):
    """
    One program per mu (program_id(0) is mu index). Iterates k=0..K-1,
    builds the 3x3 transfer entries for the current diagonal, does v <- T_k v,
    L1 renormalization, and accumulates (log c_k)/K.
    """
    b = tl.program_id(0)
    if b >= B:
        return

    mu = tl.load(MU + b)

    # local state vector v in registers (float32)
    v0 = 1.0  # v[M]
    v1 = 1.0  # v[D]
    v2 = 1.0  # v[I]
    tiny = 1e-30
    P = 0.0  # accumulate (log c_k)/K

    # loop over diagonals
    k = 0
    while k < K:
        lme_k = tl.load(LME + k)
        aD_k = tl.load(aD + k)
        bD_k = tl.load(bD + k)
        aI_k = tl.load(aI + k)
        bI_k = tl.load(bI + k)

        # E_a = exp(LME + mu)
        Ea = tl.exp(lme_k + mu)
        T_MD = tl.exp(gamma * aD_k)
        T_DD = tl.exp(gamma * bD_k)
        T_MI = tl.exp(gamma * aI_k)
        T_II = tl.exp(gamma * bI_k)

        # matvec: v' = T * v
        # M' = Ea*vM + T_MD*vD + T_MI*vI
        # D' = Ea*vM + T_DD*vD
        # I' = Ea*vM + T_II*vI
        n0 = Ea * v0 + T_MD * v1 + T_MI * v2
        n1 = Ea * v0 + T_DD * v1
        n2 = Ea * v0 + T_II * v2

        # L1 renorm
        c = tl.abs(n0) + tl.abs(n1) + tl.abs(n2)
        c = tl.maximum(c, tiny)
        n0 = n0 / c
        n1 = n1 / c
        n2 = n2 / c
        # accumulate pressure contribution
        P += tl.log(c) / K

        v0 = n0
        v1 = n1
        v2 = n2
        k += 1

    tl.store(OUT_P + b, P)


# --------------------------- Convenience API ---------------------------


@torch.no_grad()
def compute_diagonal_stats(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    gamma: float = 1.0,
    band: Optional[Tuple[int, int] | torch.Tensor] = None,
    block: int = 256,
):
    """
    Compute diagonal statistics for transfer-matrix construction on GPU.
    Inputs
      S:        (m,n) scores (log-domain, without mu)
      gD_open:  (m,) row deletion open
      gD_ext:   (m,) row deletion extend
      gI_open:  (n,) col insertion open
      gI_ext:   (n,) col insertion extend
      gamma:    tilt (>=0)
      band:     None | (jmin,jmax) | boolean (m,n) mask
    Returns dict with:
      'LME','aD','bD','aI','bI' : each (K,), where K=m+n-1
      'K' : int, 'm','n','gamma' : meta
    """
    assert S.is_cuda, "S must be on CUDA device."
    device, dtype = S.device, S.dtype
    m, n = S.shape
    jmin, jmax, S = _normalize_band(S, band)
    K = m + n - 1

    # outputs
    LME = torch.empty((K,), device=device, dtype=torch.float32)
    aD = torch.empty_like(LME)
    bD = torch.empty_like(LME)
    aI = torch.empty_like(LME)
    bI = torch.empty_like(LME)

    grid = (K,)
    diag_stats_kernel[grid](S, gD_open, gD_ext, gI_open, gI_ext, m, n, float(gamma), int(jmin), int(jmax), LME, aD, bD, aI, bI, K, BLOCK=block)

    # Cast back to S dtype for consistency if desired
    LME = LME.to(dtype)
    aD = aD.to(dtype)
    bD = bD.to(dtype)
    aI = aI.to(dtype)
    bI = bI.to(dtype)
    return {"LME": LME, "aD": aD, "bD": bD, "aI": aI, "bI": bI, "K": K, "m": m, "n": n, "gamma": float(gamma)}


@torch.no_grad()
def pressure_mu(
    stats: dict,
    mu: torch.Tensor | float,
):
    """
    Evaluate finite-length pressure P_K(mu) for a batched set of mu on GPU.
    Inputs
      stats: dict from compute_diagonal_stats
      mu:    () or (B,) tensor on CUDA (any dtype castable to float32)
    Returns
      P: (B,) tensor (or scalar) with pressure per mu
    """
    device = stats["LME"].device
    LME = stats["LME"].to(torch.float32)
    aD = stats["aD"].to(torch.float32)
    bD = stats["bD"].to(torch.float32)
    aI = stats["aI"].to(torch.float32)
    bI = stats["bI"].to(torch.float32)
    K = int(stats["K"])
    gamma = float(stats["gamma"])

    if torch.is_tensor(mu):
        MU = mu.to(device=device, dtype=torch.float32).contiguous()
    else:
        MU = torch.tensor([float(mu)], device=device, dtype=torch.float32)
    B = MU.numel()

    OUT = torch.empty((B,), device=device, dtype=torch.float32)

    grid = (B,)
    pressure_kernel[grid](
        LME,
        aD,
        bD,
        aI,
        bI,
        K,
        gamma,
        MU,
        B,
        OUT,
    )
    return OUT if torch.is_tensor(mu) else OUT[0]


@torch.no_grad()
def logZ_mu(
    stats: dict,
    mu: torch.Tensor | float,
):
    """
    Return finite-length logZ(mu) = K * P_K(mu).
    This is consistent with the renormalized power iteration accumulation.
    """
    P = pressure_mu(stats, mu)
    K = int(stats["K"])
    return P * K


@torch.no_grad()
def find_mu_c(
    stats: dict,
    mu_L: float,
    mu_U: float,
    tol: float = 1e-6,
    grid_batch: int = 128,
    max_refines: int = 4,
):
    """
    Find mu_c such that pressure P_K(mu_c) = 0 using GPU-batched bracketing / bisection-refine.
    Steps:
      1) Coarse grid over [mu_L, mu_U] with B=grid_batch -> find sign change bracket [L,U]
      2) Refine by repeatedly subdividing the bracket with B points and shrinking it.
    """
    device = stats["LME"].device
    L, U = float(mu_L), float(mu_U)

    # quick guard: ensure P(L)<0 < P(U) by expanding if needed
    def P_single(x: float) -> float:
        return float(pressure_mu(stats, torch.tensor([x], device=device)))

    PL = P_single(L)
    PU = P_single(U)
    expand = 0
    while not (PL < 0.0 < PU) and expand < 6:
        span = U - L
        L -= span
        U += span
        PL = P_single(L)
        PU = P_single(U)
        expand += 1
    if not (PL < 0.0 < PU):
        raise RuntimeError(f"Could not bracket mu_c: P({L})={PL:.6g}, P({U})={PU:.6g}")

    # multi-resolution refine
    for _ in range(max_refines):
        # dense grid inside [L,U]
        grid = torch.linspace(L, U, steps=grid_batch, device=device)
        P = pressure_mu(stats, grid)
        # find first sign change
        sign = torch.sign(P)
        # ensure not all same sign
        diffs = sign[:-1] * sign[1:]
        idx = (diffs < 0).nonzero(as_tuple=False)
        if idx.numel() == 0:
            # fallback: take the point nearest to zero and bisect there
            k = torch.argmin(torch.abs(P)).item()
            if k == 0:
                L, U = float(grid[0].item()), float(grid[1].item())
            else:
                L, U = float(grid[k - 1].item()), float(grid[k].item())
        else:
            k = int(idx[0].item())
            L, U = float(grid[k].item()), float(grid[k + 1].item())

        if abs(U - L) <= tol:
            return 0.5 * (L + U)

    # final scalar bisection
    for _ in range(60):
        M = 0.5 * (L + U)
        PM = P_single(M)
        if PM < 0.0:
            L = M
        else:
            U = M
        if abs(U - L) <= tol:
            break
    return 0.5 * (L + U)


@torch.no_grad()
def find_mu_peak_fd(
    stats: dict,
    mu_center: Optional[float] = None,
    span: float = 2.0,
    points: int = 129,
    h: float = 0.1,
):
    """
    Find mu_peak (max variance wrt mu) via finite-difference over a dense mu grid.
    We approximate:
        Var[N_M](mu) ≈ d^2/dmu^2 logZ(mu)
    using a 5-point stencil locally on the grid.
    Args
      mu_center: center of the scan; if None, it first estimates mu_c and uses that as center.
      span:      total width of the scan interval around mu_center.
      points:    number of mu samples in the scan (>=5).
      h:         delta used in the 5-point stencil (should match grid spacing for best results).
    Returns
      (mu_peak, mu_grid, logZ_grid, var_grid)
    """
    device = stats["LME"].device

    if mu_center is None:
        # rough center via small bracket around 0
        mu_center = find_mu_c(stats, -4.0, 4.0, tol=1e-3, grid_batch=128, max_refines=2)

    low = mu_center - span / 2
    high = mu_center + span / 2
    points = max(points, 5)
    mu_grid = torch.linspace(low, high, steps=points, device=device)
    logZ = logZ_mu(stats, mu_grid)  # (points,)

    # 5-point stencil for second derivative (central):
    # f''(x) ≈ (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12 h^2)
    # We'll estimate on interior indices where we have neighbors
    var = torch.full_like(logZ, float("nan"))
    # pick step as mean spacing if not exact h
    if points > 1:
        dx = float((high - low) / (points - 1))
    else:
        dx = h
    hh = h if abs(h - dx) < 1e-6 else dx

    for t in range(2, points - 2):
        fmm = logZ[t - 2]
        fm = logZ[t - 1]
        f0 = logZ[t]
        fp = logZ[t + 1]
        fpp = logZ[t + 2]
        var[t] = (-fpp + 16 * fp - 30 * f0 + 16 * fm - fmm) / (12.0 * hh * hh)

    # choose peak as argmax of variance (ignore NaN edges)
    valid = torch.isfinite(var)
    k = torch.argmax(torch.where(valid, var, torch.tensor(-1e38, device=device, dtype=var.dtype))).item()
    mu_peak = float(mu_grid[k].item())
    return mu_peak, mu_grid, logZ, var


# --------------------------- Batched Versions ---------------------------


@triton.jit
def diag_stats_kernel_batched(
    S,  # (B, m, n)
    gDo,
    gDe,  # (B, m), (B, m)
    gIo,
    gIe,  # (B, n), (B, n)
    lens1,
    lens2,  # (B,), (B,)
    LME,
    aD,
    bD,
    aI,
    bI,  # (B, K) each
    m,
    n,  # max m, n (ints)
    gamma,  # float
    jmin,
    jmax,  # ints
    K,  # int
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.program_id(1)

    m_b = tl.load(lens1 + b)
    n_b = tl.load(lens2 + b)

    i_min = tl.maximum(0, k - (n_b - 1))
    i_max = tl.minimum(k, m_b - 1)
    i_min = tl.maximum(i_min, k - jmax)
    i_max = tl.minimum(i_max, k - jmin)
    Lk = i_max - i_min + 1

    if Lk <= 0:
        if k < K:
            base = b * K
            tl.store(LME + base + k, -1e9)
            tl.store(aD + base + k, 0.0)
            tl.store(bD + base + k, 0.0)
            tl.store(aI + base + k, 0.0)
            tl.store(bI + base + k, 0.0)
        return

    # Pass 1
    t0 = 0
    xmax = tl.full((), -1e30, tl.float32)
    while t0 < Lk:
        offs = t0 + tl.arange(0, BLOCK)
        mask = offs < Lk
        i = i_min + offs
        j = k - i
        mask_valid = mask & (i < m_b) & (j < n_b)  # Add per-batch bounds
        offset = tl.where(mask_valid, i * n + j, 0)
        sij = tl.load(S + b * m * n + offset, mask=mask_valid, other=-1e9)
        x = gamma * sij
        x = tl.where(mask_valid, x, -1e30)
        xmax = tl.maximum(xmax, tl.max(x, axis=0))
        t0 += BLOCK

    # Pass 2
    t0 = 0
    sumexp = tl.zeros((), dtype=tl.float32)
    sum_gDo = tl.zeros((), dtype=tl.float32)
    sum_gDe = tl.zeros((), dtype=tl.float32)
    sum_gIo = tl.zeros((), dtype=tl.float32)
    sum_gIe = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    while t0 < Lk:
        offs = t0 + tl.arange(0, BLOCK)
        mask = offs < Lk
        i = i_min + offs
        j = k - i
        mask_valid = mask & (i < m_b) & (j < n_b)
        offset = tl.where(mask_valid, i * n + j, 0)
        sij = tl.load(S + b * m * n + offset, mask=mask_valid, other=-1e9)
        x = gamma * sij
        expx = tl.exp(x - xmax)
        sumexp += tl.sum(tl.where(mask_valid, expx, 0.0), axis=0)
        i_offset = tl.where(mask_valid, i, 0)
        j_offset = tl.where(mask_valid, j, 0)
        gd_o = tl.load(gDo + b * m + i_offset, mask=mask_valid, other=0.0)
        gd_e = tl.load(gDe + b * m + i_offset, mask=mask_valid, other=0.0)
        gi_o = tl.load(gIo + b * n + j_offset, mask=mask_valid, other=0.0)
        gi_e = tl.load(gIe + b * n + j_offset, mask=mask_valid, other=0.0)
        sum_gDo += tl.sum(tl.where(mask_valid, gd_o, 0.0), axis=0)
        sum_gDe += tl.sum(tl.where(mask_valid, gd_e, 0.0), axis=0)
        sum_gIo += tl.sum(tl.where(mask_valid, gi_o, 0.0), axis=0)
        sum_gIe += tl.sum(tl.where(mask_valid, gi_e, 0.0), axis=0)
        count += tl.sum(tl.where(mask_valid, 1.0, 0.0), axis=0)
        t0 += BLOCK

    # finalize
    lme = xmax + tl.log(sumexp / tl.maximum(count, 1.0))
    mean_gDo = sum_gDo / tl.maximum(count, 1.0)
    mean_gDe = sum_gDe / tl.maximum(count, 1.0)
    mean_gIo = sum_gIo / tl.maximum(count, 1.0)
    mean_gIe = sum_gIe / tl.maximum(count, 1.0)

    base = b * K
    tl.store(LME + base + k, lme)
    tl.store(aD + base + k, mean_gDo)
    tl.store(bD + base + k, mean_gDe)
    tl.store(aI + base + k, mean_gIo)
    tl.store(bI + base + k, mean_gIe)


@torch.no_grad()
def compute_diagonal_stats_batched(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    lens1: torch.Tensor,
    lens2: torch.Tensor,
    gamma: float = 1.0,
    band: Optional[Tuple[int, int] | torch.Tensor] = None,
    block: int = 256,
):
    """
    Batched version of compute_diagonal_stats.
    Inputs
      S:        (B, m, n) scores
      gD_open:  (B, m)
      gD_ext:   (B, m)
      gI_open:  (B, n)
      gI_ext:   (B, n)
      lens1:    (B,) actual lengths for dim 1
      lens2:    (B,) actual lengths for dim 2
      gamma:    tilt
    Returns dict with batched stats:
      'LME','aD','bD','aI','bI' : each (B, K)
    """
    assert S.is_cuda, "S must be on CUDA device."
    assert S.ndim == 3, "S must be a 3D tensor (B, m, n)"
    device = S.device
    B, m, n = S.shape
    K = m + n - 1  # Max number of anti-diagonals

    # Normalize band (assuming shared; if band is tensor mask, apply per-batch)
    if band is None:
        jmin, jmax = 0, n - 1
    else:
        jmin, jmax, _ = _normalize_band(S[0], band)  # Placeholder; extend for batch if needed

    # Allocate outputs
    LME = torch.empty((B, K), device=device, dtype=torch.float32)
    aD = torch.empty_like(LME)
    bD = torch.empty_like(LME)
    aI = torch.empty_like(LME)
    bI = torch.empty_like(LME)

    # Launch kernel with 2D grid: program_id(0)=b (batch), program_id(1)=k (diagonal)
    grid = (B, K)
    diag_stats_kernel_batched[grid](S, gD_open, gD_ext, gI_open, gI_ext, lens1, lens2, LME, aD, bD, aI, bI, m, n, gamma, jmin, jmax, K, BLOCK=block)

    return {"LME": LME, "aD": aD, "bD": bD, "aI": aI, "bI": bI, "K": K, "gamma": gamma, "m": m, "n": n}


@triton.jit
def pressure_kernel_batched(
    # stats over k, batched
    LME,
    aD,
    bD,
    aI,
    bI,  # (B, K)
    K,
    gamma,
    # mu-batch
    MU,  # (B,)
    B,
    # outputs
    OUT_P,  # (B,)
):
    """
    Batched version of pressure_kernel. One program per mu, but stats are also batched.
    """
    b = tl.program_id(0)
    if b >= B:
        return

    mu = tl.load(MU + b)

    v0, v1, v2 = 1.0, 1.0, 1.0
    tiny = 1e-30
    P = 0.0

    k = 0
    while k < K:
        stats_offset = b * K + k
        lme_k = tl.load(LME + stats_offset)
        aD_k = tl.load(aD + stats_offset)
        bD_k = tl.load(bD + stats_offset)
        aI_k = tl.load(aI + stats_offset)
        bI_k = tl.load(bI + stats_offset)

        Ea = tl.exp(lme_k + mu)
        T_MD = tl.exp(gamma * aD_k)
        T_DD = tl.exp(gamma * bD_k)
        T_MI = tl.exp(gamma * aI_k)
        T_II = tl.exp(gamma * bI_k)

        n0 = Ea * v0 + T_MD * v1 + T_MI * v2
        n1 = Ea * v0 + T_DD * v1
        n2 = Ea * v0 + T_II * v2

        c = tl.abs(n0) + tl.abs(n1) + tl.abs(n2)
        c = tl.maximum(c, tiny)
        n0, n1, n2 = n0 / c, n1 / c, n2 / c
        P += tl.log(c) / K
        v0, v1, v2 = n0, n1, n2
        k += 1

    tl.store(OUT_P + b, P)


@torch.no_grad()
def pressure_mu_batched(stats: dict, mu: torch.Tensor):
    """
    Evaluates pressure for batched stats and batched mu.
    Inputs
      stats: dict from compute_diagonal_stats_batched, with shapes (B, K)
      mu:    (B,) or (B, N_mu) tensor on CUDA
    Returns
      P: (B,) or (B, N_mu) tensor with pressure per mu
    """
    device = stats["LME"].device
    B, K = stats["LME"].shape
    gamma = float(stats["gamma"])

    if mu.ndim == 1:  # mu is (B,)
        assert mu.shape[0] == B
        MU = mu.to(device=device, dtype=torch.float32).contiguous()
        LME, aD, bD, aI, bI = (
            stats["LME"].to(torch.float32),
            stats["aD"].to(torch.float32),
            stats["bD"].to(torch.float32),
            stats["aI"].to(torch.float32),
            stats["bI"].to(torch.float32),
        )
        OUT = torch.empty((B,), device=device, dtype=torch.float32)
        grid = (B,)
        pressure_kernel_batched[grid](LME, aD, bD, aI, bI, K, gamma, MU, B, OUT)
        return OUT

    elif mu.ndim == 2:  # mu is (B, N_mu)
        _B, N_mu = mu.shape
        assert _B == B
        mu_flat = mu.reshape(-1)
        B_flat = mu_flat.numel()

        LME = stats["LME"].repeat_interleave(N_mu, dim=0).to(torch.float32)
        aD = stats["aD"].repeat_interleave(N_mu, dim=0).to(torch.float32)
        bD = stats["bD"].repeat_interleave(N_mu, dim=0).to(torch.float32)
        aI = stats["aI"].repeat_interleave(N_mu, dim=0).to(torch.float32)
        bI = stats["bI"].repeat_interleave(N_mu, dim=0).to(torch.float32)

        OUT_flat = torch.empty((B_flat,), device=device, dtype=torch.float32)
        grid = (B_flat,)
        pressure_kernel_batched[grid](LME, aD, bD, aI, bI, K, gamma, mu_flat, B_flat, OUT_flat)
        return OUT_flat.reshape(B, N_mu)

    raise ValueError("mu must be a 1D or 2D tensor")


@torch.no_grad()
def logZ_mu_batched(stats: dict, mu: torch.Tensor):
    P = pressure_mu_batched(stats, mu)
    K = int(stats["K"])
    return P * K


@torch.no_grad()
def find_mu_c_batched(
    stats: dict,
    mu_L: float,
    mu_U: float,
    tol: float = 1e-6,
    grid_batch: int = 128,
    max_refines: int = 4,
):
    """Batched version of find_mu_c."""
    device = stats["LME"].device
    B = stats["LME"].shape[0]

    L = torch.full((B,), mu_L, device=device, dtype=torch.float32)
    U = torch.full((B,), mu_U, device=device, dtype=torch.float32)

    # Guard for bracket
    PL = pressure_mu_batched(stats, L)
    PU = pressure_mu_batched(stats, U)
    needs_expansion = ~((PL < 0.0) & (PU > 0.0))
    expand_iter = 0
    while needs_expansion.any() and expand_iter < 6:
        span = U - L
        L[needs_expansion] -= span[needs_expansion]
        U[needs_expansion] += span[needs_expansion]
        PL = pressure_mu_batched(stats, L)
        PU = pressure_mu_batched(stats, U)
        needs_expansion = ~((PL < 0.0) & (PU > 0.0))
        expand_iter += 1

    if needs_expansion.any():
        failed_indices = needs_expansion.nonzero().squeeze().tolist()
        print(f"WARNING: Could not bracket mu_c for indices: {failed_indices}")
        # For failed items, default to a small value and continue
        L[needs_expansion] = -1.0
        U[needs_expansion] = 1.0

    # Multi-resolution refine
    for _ in range(max_refines):
        grid = torch.linspace(0, 1, steps=grid_batch, device=device)[None, :] * (U - L)[:, None] + L[:, None]
        P = pressure_mu_batched(stats, grid)
        sign = torch.sign(P)
        diffs = sign[:, :-1] * sign[:, 1:]

        # Find first sign change for each batch item
        found_change = (diffs < 0).any(dim=1)
        idx = torch.argmax((diffs < 0).int(), dim=1)

        # For those with no sign change, pick point closest to zero
        no_change_mask = ~found_change
        if no_change_mask.any():
            k_fallback = torch.argmin(torch.abs(P[no_change_mask]), dim=1)

            # Update L and U for fallback cases
            grid_no_change = grid[no_change_mask]
            num_no_change = grid_no_change.shape[0]
            arange_no_change = torch.arange(num_no_change, device=device)

            L_fallback = grid_no_change[arange_no_change, (k_fallback - 1).clamp(min=0)]
            U_fallback = grid_no_change[arange_no_change, k_fallback]

            L[no_change_mask] = L_fallback
            U[no_change_mask] = U_fallback

        # Update L and U for cases with sign change
        if found_change.any():
            L[found_change] = grid[found_change, idx[found_change]]
            U[found_change] = grid[found_change, idx[found_change] + 1]

        if ((U - L).abs() <= tol).all():
            break

    # Final scalar bisection (vectorized)
    for _ in range(60):
        M = 0.5 * (L + U)
        PM = pressure_mu_batched(stats, M)
        L = torch.where(PM < 0.0, M, L)
        U = torch.where(PM < 0.0, U, M)
        if ((U - L).abs() <= tol).all():
            break

    return 0.5 * (L + U)


@torch.no_grad()
def find_mu_peak_fd_batched(
    stats: dict,
    mu_centers: torch.Tensor,
    span: float = 20.0,
    points: int = 255,
    h: float = 0.1,
):
    """Batched version of find_mu_peak_fd."""
    device = stats["LME"].device
    B = stats["LME"].shape[0]
    assert mu_centers.shape == (B,)

    lows = mu_centers - span / 2
    highs = mu_centers + span / 2
    points = max(points, 5)

    # Create a grid for each item in the batch
    mu_grid = torch.linspace(0, 1, steps=points, device=device)[None, :] * (highs - lows)[:, None] + lows[:, None]  # (B, points)
    logZ = logZ_mu_batched(stats, mu_grid)  # (B, points)

    # 5-point stencil for second derivative
    var = torch.full_like(logZ, float("nan"))
    dx = (highs - lows) / (points - 1)
    hh = dx  # Use the calculated grid spacing for each batch item

    # Apply stencil in a vectorized way
    fmm = logZ[:, :-4]
    fm = logZ[:, 1:-3]
    f0 = logZ[:, 2:-2]
    fp = logZ[:, 3:-1]
    fpp = logZ[:, 4:]

    var[:, 2:-2] = (-fpp + 16 * fp - 30 * f0 + 16 * fm - fmm) / (12.0 * hh[:, None] * hh[:, None])

    # Argmax over the valid (non-NaN) part of the variance
    valid_var = torch.where(torch.isfinite(var), var, -1e38)
    k = torch.argmax(valid_var, dim=1)

    mu_peaks = mu_grid[torch.arange(B), k]

    return mu_peaks, mu_grid, logZ, var


# --------------------------- Example ---------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    m, n = 800, 900

    # Mock data (log-scores and gap parameters)
    S = 0.2 * torch.randn(m, n, device=device)
    gDo = torch.full((m,), -1.2, device=device)
    gDe = torch.full((m,), -0.3, device=device)
    gIo = torch.full((n,), -1.2, device=device)
    gIe = torch.full((n,), -0.3, device=device)
    gamma = 1.0

    # 1) Diagonal stats
    stats = compute_diagonal_stats(S, gDo, gDe, gIo, gIe, gamma=gamma, band=None, block=256)

    # 2) Pressure for a batched mu
    mu_batch = torch.linspace(-2.0, 2.0, steps=17, device=device)
    P = pressure_mu(stats, mu_batch)
    print("Pressure grid:", P.detach().cpu().numpy())

    # 3) Find mu_c (P=0)
    mu_c = find_mu_c(stats, -4.0, 4.0, tol=1e-5, grid_batch=129, max_refines=3)
    print("mu_c ≈", mu_c)

    # 4) Find mu_peak via FD around mu_c
    mu_peak, mu_grid, logZ_grid, var_grid = find_mu_peak_fd(stats, mu_center=mu_c, span=2.0, points=129, h=0.1)
    print("mu_peak ≈", mu_peak)
