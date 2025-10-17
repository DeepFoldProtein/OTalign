# softdp_triton_full.py
# 3-state alignment CRF (M/D/I) soft-DP with anti-diagonal (wavefront) schedule.
# Both FORWARD and BACKWARD passes are implemented as exact Triton kernels.
# A convenience API returns the six matrices + logZ (global or local).
#
# States:
#   M: match  (enter adds S[i,j] + mu)
#   D: deletion (gap in B)   with row-wise gaps gD_open[i], gD_ext[i]
#   I: insertion (gap in A)  with col-wise gaps gI_open[j], gI_ext[j]
#
# Recurrences (log-domain):
#   F_M[i,j] = (S[i,j] + mu) + LSE(F_M[i-1,j-1], F_D[i-1,j-1], F_I[i-1,j-1])
#   F_D[i,j] = LSE(F_M[i-1,j] + gD_open[i], F_D[i-1,j] + gD_ext[i])
#   F_I[i,j] = LSE(F_M[i,j-1] + gI_open[j], F_I[i,j-1] + gI_ext[j])
#
#   B_M[i,j] = LSE( (S[i+1,j+1]+mu) + B_M[i+1,j+1], gD_open[i+1] + B_D[i+1,j], gI_open[j+1] + B_I[i,j+1] )
#   B_D[i,j] = LSE( (S[i+1,j+1]+mu) + B_M[i+1,j+1], gD_ext[i+1]  + B_D[i+1,j] )
#   B_I[i,j] = LSE( (S[i+1,j+1]+mu) + B_M[i+1,j+1], gI_ext[j+1]  + B_I[i,j+1] )
#

import torch
import triton
import triton.language as tl


def _neg_inf(dtype: torch.dtype) -> float:
    finfo = torch.finfo(dtype)
    return float(finfo.min / 4.0)


def _as_band(S: torch.Tensor, band) -> tuple[int, int, torch.Tensor]:
    """
    Normalize band spec into (jmin, jmax, S_masked).
    If band is a boolean mask, set S[~mask] = -inf and return full j-range.
    """
    m, n = S.shape
    NEG = _neg_inf(S.dtype)
    if band is None:
        return 0, n - 1, S
    if isinstance(band, tuple) and len(band) == 2:
        jmin, jmax = int(band[0]), int(band[1])
        return jmin, jmax, S
    if torch.is_tensor(band) and band.dtype == torch.bool and band.shape == S.shape:
        S2 = torch.where(band, S, torch.full_like(S, NEG))
        return 0, n - 1, S2
    raise ValueError("band must be None, (jmin,jmax), or boolean mask (m,n)")


def _anti_diag_bounds(k: int, m: int, n: int, jmin: int, jmax: int):
    """
    For anti-diagonal k (0..m+n-2), return (i_min, i_max, Lk).
    We enforce the column band j in [jmin..jmax] by clamping i-range accordingly.
    """
    i_min = max(0, k - (n - 1))
    i_max = min(k, m - 1)
    i_min = max(i_min, k - jmax)
    i_max = min(i_max, k - jmin)
    Lk = max(0, i_max - i_min + 1)
    return i_min, i_max, Lk


# -------------------------- Triton kernels --------------------------


@triton.jit
def fwd_diag_kernel(
    # prev-2 diagonal parents (for M):
    FM_km2,
    FD_km2,
    FI_km2,  # (B, L2)
    # prev-1 diagonal parents (for D/I):
    FM_km1,
    FD_km1,
    FI_km1,  # (B, L1)
    # params
    S_M,  # (B, m, n) = S + mu
    gDo,
    gDe,
    gIo,
    gIe,  # (m,), (m,), (n,), (n,)
    # outputs for current diagonal
    FM_k,
    FD_k,
    FI_k,  # (B, Lk)
    # meta
    i_min,
    k,
    Lk,
    i1_min,
    L1,
    i2_min,
    L2,
    m,
    n,
    NEG,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # tile id along current diagonal
    b = tl.program_id(1)  # batch id (mu-batch)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Lk

    i = i_min + offs
    j = k - i
    valid = mask & (i >= 0) & (i < m) & (j >= 0) & (j < n)

    # S+mu at current cells
    s_ij = tl.load(S_M + b * m * n + i * n + j, mask=valid, other=NEG)

    # ----- M from k-2: parents at (i-1,j-1) on diag k-2 -----
    tM = (i - 1) - i2_min
    has_km2 = L2 > 0
    maskM = valid & has_km2 & (tM >= 0) & (tM < L2)
    base2 = b * L2
    fm2 = tl.load(FM_km2 + base2 + tM, mask=maskM, other=NEG)
    fd2 = tl.load(FD_km2 + base2 + tM, mask=maskM, other=NEG)
    fi2 = tl.load(FI_km2 + base2 + tM, mask=maskM, other=NEG)
    mx2 = tl.maximum(fm2, tl.maximum(fd2, fi2))
    lse2 = tl.where(mx2 > NEG, mx2 + tl.log(tl.exp(fm2 - mx2) + tl.exp(fd2 - mx2) + tl.exp(fi2 - mx2)), NEG)
    FMv = tl.where(maskM & (lse2 > NEG), s_ij + lse2, NEG)

    # ----- D from k-1: parents at (i-1,j) on diag k-1 -----
    tD = (i - 1) - i1_min
    has_km1 = L1 > 0
    maskD = valid & has_km1 & (tD >= 0) & (tD < L1)
    base1 = b * L1
    fm1D = tl.load(FM_km1 + base1 + tD, mask=maskD, other=NEG)
    fd1D = tl.load(FD_km1 + base1 + tD, mask=maskD, other=NEG)
    gDo_i = tl.load(gDo + i, mask=maskD, other=0.0)
    gDe_i = tl.load(gDe + i, mask=maskD, other=0.0)
    cD1 = fm1D + gDo_i
    cD2 = fd1D + gDe_i
    mxD = tl.maximum(cD1, cD2)
    lseD = tl.where(mxD > NEG, mxD + tl.log(tl.exp(cD1 - mxD) + tl.exp(cD2 - mxD)), NEG)
    FDv = tl.where(maskD, lseD, NEG)

    # ----- I from k-1: parents at (i,j-1) on diag k-1 -----
    tI = i - i1_min
    maskI = valid & has_km1 & (tI >= 0) & (tI < L1)
    fm1I = tl.load(FM_km1 + base1 + tI, mask=maskI, other=NEG)
    fi1I = tl.load(FI_km1 + base1 + tI, mask=maskI, other=NEG)
    gIo_j = tl.load(gIo + j, mask=maskI, other=0.0)
    gIe_j = tl.load(gIe + j, mask=maskI, other=0.0)
    cI1 = fm1I + gIo_j
    cI2 = fi1I + gIe_j
    mxI = tl.maximum(cI1, cI2)
    lseI = tl.where(mxI > NEG, mxI + tl.log(tl.exp(cI1 - mxI) + tl.exp(cI2 - mxI)), NEG)
    FIv = tl.where(maskI, lseI, NEG)

    # store
    tl.store(FM_k + b * Lk + offs, tl.where(mask, FMv, NEG))
    tl.store(FD_k + b * Lk + offs, tl.where(mask, FDv, NEG))
    tl.store(FI_k + b * Lk + offs, tl.where(mask, FIv, NEG))


@triton.jit
def bwd_diag_kernel(
    # successors:
    BM_kp2,
    BD_kp2,
    BI_kp2,  # (B, L2) for k+2
    BM_kp1,
    BD_kp1,
    BI_kp1,  # (B, L1) for k+1
    # params
    S_M,  # (B, m, n) = S + mu
    gDo,
    gDe,
    gIo,
    gIe,  # (m,), (m,), (n,), (n,)
    # outputs for current diagonal
    BM_k,
    BD_k,
    BI_k,  # (B, Lk)
    # meta
    i_min,
    k,
    Lk,
    i1_min,
    L1,
    i2_min,
    L2,
    m,
    n,
    NEG,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Lk

    i = i_min + offs
    j = k - i
    valid = mask & (i >= 0) & (i < m) & (j >= 0) & (j < n)

    # Clamp indices to avoid out-of-bounds pointer arithmetic, even if masked.
    ip1 = i + 1
    jp1 = j + 1
    safe_ip1 = tl.where(ip1 < m, ip1, 0)
    safe_jp1 = tl.where(jp1 < n, jp1, 0)

    # --- Successor values ---
    # Successor (i+1, j+1) on diag k+2
    tM_succ = ip1 - i2_min
    has_kp2 = L2 > 0
    mask_M_succ = valid & has_kp2 & (tM_succ >= 0) & (tM_succ < L2) & (ip1 < m) & (jp1 < n)
    s_m_succ = tl.load(S_M + b * m * n + safe_ip1 * n + safe_jp1, mask=mask_M_succ, other=NEG)
    bm_succ = tl.load(BM_kp2 + b * L2 + tM_succ, mask=mask_M_succ, other=NEG)

    # Successor (i+1, j) on diag k+1
    tD_succ = ip1 - i1_min
    has_kp1 = L1 > 0
    mask_D_succ = valid & has_kp1 & (tD_succ >= 0) & (tD_succ < L1) & (ip1 < m)
    bd_succ = tl.load(BD_kp1 + b * L1 + tD_succ, mask=mask_D_succ, other=NEG)

    # Successor (i, j+1) on diag k+1
    tI_succ = i - i1_min
    mask_I_succ = valid & has_kp1 & (tI_succ >= 0) & (tI_succ < L1) & (jp1 < n)
    bi_succ = tl.load(BI_kp1 + b * L1 + tI_succ, mask=mask_I_succ, other=NEG)

    # --- Gap scores for transitions ---
    gDo_ip1 = tl.load(gDo + safe_ip1, mask=mask_D_succ, other=0.0)
    gDe_ip1 = tl.load(gDe + safe_ip1, mask=mask_D_succ, other=0.0)
    gIo_jp1 = tl.load(gIo + safe_jp1, mask=mask_I_succ, other=0.0)
    gIe_jp1 = tl.load(gIe + safe_jp1, mask=mask_I_succ, other=0.0)

    # --- Terms for LSE ---
    # Term for M->M, D->M, I->M transition: (S[i+1,j+1]+mu) + B_M[i+1,j+1]
    term_m_succ = tl.where(mask_M_succ, s_m_succ + bm_succ, NEG)

    # Term for M->D transition: gD_open[i+1] + B_D[i+1,j]
    term_d_open = tl.where(mask_D_succ, gDo_ip1 + bd_succ, NEG)
    # Term for D->D transition: gD_ext[i+1] + B_D[i+1,j]
    term_d_ext = tl.where(mask_D_succ, gDe_ip1 + bd_succ, NEG)

    # Term for M->I transition: gI_open[j+1] + B_I[i,j+1]
    term_i_open = tl.where(mask_I_succ, gIo_jp1 + bi_succ, NEG)
    # Term for I->I transition: gI_ext[j+1] + B_I[i,j+1]
    term_i_ext = tl.where(mask_I_succ, gIe_jp1 + bi_succ, NEG)

    # --- Compute B_M, B_D, B_I ---
    # B_M[i,j] = LSE(term_m_succ, term_d_open, term_i_open)
    mx_m = tl.maximum(term_m_succ, tl.maximum(term_d_open, term_i_open))
    bm_k = tl.where(mx_m > NEG, mx_m + tl.log(tl.exp(term_m_succ - mx_m) + tl.exp(term_d_open - mx_m) + tl.exp(term_i_open - mx_m)), NEG)

    # B_D[i,j] = LSE(term_m_succ, term_d_ext)
    mx_d = tl.maximum(term_m_succ, term_d_ext)
    bd_k = tl.where(mx_d > NEG, mx_d + tl.log(tl.exp(term_m_succ - mx_d) + tl.exp(term_d_ext - mx_d)), NEG)

    # B_I[i,j] = LSE(term_m_succ, term_i_ext)
    mx_i = tl.maximum(term_m_succ, term_i_ext)
    bi_k = tl.where(mx_i > NEG, mx_i + tl.log(tl.exp(term_m_succ - mx_i) + tl.exp(term_i_ext - mx_i)), NEG)

    # --- Store results ---
    tl.store(BM_k + b * Lk + offs, tl.where(valid, bm_k, NEG))
    tl.store(BD_k + b * Lk + offs, tl.where(valid, bd_k, NEG))
    tl.store(BI_k + b * Lk + offs, tl.where(valid, bi_k, NEG))


# -------------------------- Triton drivers + API --------------------------


@torch.no_grad()
def softdp_forward_triton(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    mu: torch.Tensor | float,
    *,
    band: tuple[int, int] | torch.Tensor | None = None,
    block_size: int = 128,
):
    """
    Run FORWARD soft-DP (exact) with Triton kernels. Returns (F_M,F_D,F_I, logZ).
    Shapes:
      S: (m,n), gaps: gD_* (m,), gI_* (n,), mu: () or (B,)
      Outputs: F_* : (B,m,n) if batched, else (m,n); logZ: (B,) or scalar
    """
    device, dtype = S.device, S.dtype
    NEG = _neg_inf(dtype)
    m, n = S.shape

    jmin, jmax, S = _as_band(S, band)

    # μ-batch handling
    batched = torch.is_tensor(mu) and getattr(mu, "ndim", 0) == 1
    mu_vec = mu if batched else torch.tensor([float(mu)], device=device, dtype=dtype)
    B = mu_vec.numel()

    # broadcast S + mu
    S_M = S.unsqueeze(0) + mu_vec[:, None, None]  # (B,m,n)

    # outputs
    FM = torch.full((B, m, n), NEG, device=device, dtype=dtype)
    FD = torch.full_like(FM, NEG)
    FI = torch.full_like(FM, NEG)

    # ring buffers for prev diagonals
    KM2 = KM1 = None

    # Handle k=0 case for initialization
    # glocal: classic boundaries with free opening gaps
    i_min_0, _, Lk_0 = _anti_diag_bounds(0, m, n, jmin, jmax)
    if Lk_0 > 0:
        # i=0, j=0 is the only cell. Can only be a match.
        # For global and glocal, F[0,0] (for state M) is S_M[0,0], as it comes from a virtual start state with score 0.
        FM[:, 0, 0] = S_M[:, 0, 0]
        FMk = S_M[:, 0, 0].unsqueeze(1)
        FDk = torch.full_like(FMk, NEG)
        FIk = torch.full_like(FMk, NEG)
        KM1 = (FMk, FDk, FIk)

    K = m + n - 1
    for k in range(1, K):
        i_min, i_max, Lk = _anti_diag_bounds(k, m, n, jmin, jmax)
        if Lk == 0:
            KM2, KM1 = KM1, None
            continue

        # neighbor diagonal bounds
        def bounds(k_):
            if k_ < 0 or k_ >= K:
                return None
            imn, imx, L = _anti_diag_bounds(k_, m, n, jmin, jmax)
            return (imn, imx, L) if L > 0 else None

        b1 = bounds(k - 1)
        b2 = bounds(k - 2)

        # allocate output slices for current diag
        FMk = torch.full((B, Lk), NEG, device=device, dtype=dtype)
        FDk = torch.full_like(FMk, NEG)
        FIk = torch.full_like(FMk, NEG)

        # prepare prev-1 and prev-2 diagonal buffers (or safe dummies)
        if b1 is not None and KM1 is not None:
            i1_min, _, L1 = b1
            FM1, FD1, FI1 = KM1
        else:
            i1_min, L1 = 0, 0
            FM1 = FD1 = FI1 = torch.empty((B, 1), device=device, dtype=dtype)

        if b2 is not None and KM2 is not None:
            i2_min, _, L2 = b2
            FM2, FD2, FI2 = KM2
        else:
            i2_min, L2 = 0, 0
            FM2 = FD2 = FI2 = torch.empty((B, 1), device=device, dtype=dtype)

        # kernel launch
        grid = ((Lk + block_size - 1) // block_size, B)
        fwd_diag_kernel[grid](FM2, FD2, FI2, FM1, FD1, FI1, S_M, gD_open, gD_ext, gI_open, gI_ext, FMk, FDk, FIk, i_min, k, Lk, i1_min, L1, i2_min, L2, m, n, NEG, BLOCK=block_size)

        # scatter diagonal slices into full grids
        t = torch.arange(Lk, device=device)

        if k > 0:
            i_coords = i_min + t
            j_coords = k - i_coords

            # Override for j=0 boundary (all i > 0)
            j_zero_mask = j_coords == 0
            if j_zero_mask.any():
                FDk[:, j_zero_mask] = 0.0

            # Override for i=0 boundary (all j > 0)
            i_zero_mask = i_coords == 0
            if i_zero_mask.any():
                FIk[:, i_zero_mask] = 0.0

        i = i_min + t
        j = k - i
        FM[:, i, j] = FMk
        FD[:, i, j] = FDk
        FI[:, i, j] = FIk

        # rotate ring
        KM2, KM1 = KM1, (FMk, FDk, FIk)

        # --- DEBUG ---
        if Lk > 0 and FMk.max() < NEG and FDk.max() < NEG and FIk.max() < NEG:
            print(f"WARNING: All scores on diagonal k={k} are NEG. Propagation has failed.")
            if KM1 is not None:
                fm1, fd1, fi1 = KM1
                print(f"  k-1 max scores: M={fm1.max().item()}, D={fd1.max().item()}, I={fi1.max().item()}")
            if KM2 is not None:
                fm2, fd2, fi2 = KM2
                print(f"  k-2 max scores: M={fm2.max().item()}, D={fd2.max().item()}, I={fi2.max().item()}")

    z = torch.stack([FM[:, -1, -1], FD[:, -1, -1], FI[:, -1, -1]], dim=0)  # (3,B)
    mx = z.max(dim=0, keepdim=True).values
    logZ = torch.logsumexp(z - mx, dim=0) + mx.squeeze(0)

    # squeeze if scalar mu
    if not batched:
        return FM.squeeze(0), FD.squeeze(0), FI.squeeze(0), float(logZ.squeeze(0))
    return FM, FD, FI, logZ


@torch.no_grad()
def softdp_backward_triton(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    mu: torch.Tensor | float,
    *,
    band: tuple[int, int] | torch.Tensor | None = None,
    block_size: int = 128,
):
    """
    Run BACKWARD soft-DP (exact) with Triton kernels. Returns (B_M,B_D,B_I).
    Shapes follow softdp_forward_triton.
    """
    device, dtype = S.device, S.dtype
    NEG = _neg_inf(dtype)
    m, n = S.shape

    jmin, jmax, S = _as_band(S, band)
    batched = torch.is_tensor(mu) and getattr(mu, "ndim", 0) == 1
    mu_vec = mu if batched else torch.tensor([float(mu)], device=device, dtype=dtype)
    B = mu_vec.numel()
    S_M = S.unsqueeze(0) + mu_vec[:, None, None]

    BM = torch.full((B, m, n), NEG, device=device, dtype=dtype)
    BD = torch.full_like(BM, NEG)
    BI = torch.full_like(BM, NEG)

    K = m + n - 1
    KP2 = KP1 = None  # future rings: k+2 and k+1

    # Handle k=K-1 for initialization
    i_min_last, _, Lk_last = _anti_diag_bounds(K - 1, m, n, jmin, jmax)
    if Lk_last > 0:
        # The last cell (m-1, n-1) is the only one on this diagonal within bounds
        # All backward variables are 0 (log(1)) at the end state.
        BM[:, m - 1, n - 1] = 0.0
        BD[:, m - 1, n - 1] = 0.0
        BI[:, m - 1, n - 1] = 0.0

        BMk = torch.zeros((B, 1), device=device, dtype=dtype)
        BDk = torch.zeros((B, 1), device=device, dtype=dtype)
        BIk = torch.zeros((B, 1), device=device, dtype=dtype)
        KP1 = (BMk, BDk, BIk)

    for k in range(K - 2, -1, -1):
        i_min, i_max, Lk = _anti_diag_bounds(k, m, n, jmin, jmax)
        if Lk == 0:
            KP2, KP1 = KP1, None
            continue

        def bounds(k_):
            if k_ < 0 or k_ >= K:
                return None
            imn, imx, L = _anti_diag_bounds(k_, m, n, jmin, jmax)
            return (imn, imx, L) if L > 0 else None

        b1 = bounds(k + 1)
        b2 = bounds(k + 2)

        BMk = torch.full((B, Lk), NEG, device=device, dtype=dtype)
        BDk = torch.full_like(BMk, NEG)
        BIk = torch.full_like(BMk, NEG)

        if b2 is not None and KP2 is not None:
            i2_min, _, L2 = b2
            BM2, BD2, BI2 = KP2
        else:
            i2_min, L2 = 0, 0
            BM2 = BD2 = BI2 = torch.empty((B, 1), device=device, dtype=dtype)

        if b1 is not None and KP1 is not None:
            i1_min, _, L1 = b1
            BM1, BD1, BI1 = KP1
        else:
            i1_min, L1 = 0, 0
            BM1 = BD1 = BI1 = torch.empty((B, 1), device=device, dtype=dtype)

        grid = ((Lk + block_size - 1) // block_size, B)
        bwd_diag_kernel[grid](BM2, BD2, BI2, BM1, BD1, BI1, S_M, gD_open, gD_ext, gI_open, gI_ext, BMk, BDk, BIk, i_min, k, Lk, i1_min, L1, i2_min, L2, m, n, NEG, BLOCK=block_size)

        # scatter into full grids
        t = torch.arange(Lk, device=device)
        i = i_min + t
        j = k - i
        BM[:, i, j] = BMk
        BD[:, i, j] = BDk
        BI[:, i, j] = BIk

        # rotate
        KP2, KP1 = KP1, (BMk, BDk, BIk)

    if not batched:
        return BM.squeeze(0), BD.squeeze(0), BI.squeeze(0)
    return BM, BD, BI


@torch.no_grad()
def forward_backward(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    mu: torch.Tensor | float,
    *,
    band: tuple[int, int] | torch.Tensor | None = None,
    block_size: int = 128,
):
    """
    Full soft-DP with Triton forward + Triton backward.
    Returns dict with the six matrices + logZ.

    Args
    ----
      S: (m,n) base match scores (log-domain, WITHOUT mu)
      gD_open, gD_ext: (m,)
      gI_open, gI_ext: (n,)
      mu: scalar or (B,) vector
      band: None | (jmin,jmax) | boolean (m,n) mask
      local: if True, logZ is LSE over all cells; else global end cell
    """
    assert S.ndim == 2, "S must be (m,n)"
    m, n = S.shape
    assert gD_open.shape == (m,) and gD_ext.shape == (m,), "row gap vectors must be (m,)"
    assert gI_open.shape == (n,) and gI_ext.shape == (n,), "col gap vectors must be (n,)"

    FM, FD, FI, logZ = softdp_forward_triton(S, gD_open, gD_ext, gI_open, gI_ext, mu, band=band, block_size=block_size)
    BM, BD, BI = softdp_backward_triton(S, gD_open, gD_ext, gI_open, gI_ext, mu, band=band, block_size=block_size)
    return {"F_M": FM, "F_D": FD, "F_I": FI, "B_M": BM, "B_D": BD, "B_I": BI, "logZ": logZ}


def find_longest_increasing_sequence_2d(grid):
    """
    2D 불리언 배열에서 최장 증가 시퀀스를 찾습니다.
    시퀀스는 i_1 < i_2 이고 j_1 < j_2 인 (row, col) 좌표의 연속으로 정의됩니다.

    Args:
        grid (list[list[bool]]): 입력 2D 불리언 배열.

    Returns:
        list[tuple[int, int]]: 최장 증가 시퀀스를 구성하는 (row, col) 좌표 리스트.
                                 시퀀스를 찾지 못한 경우 빈 리스트를 반환합니다.
    """
    # 1. 'True'인 모든 좌표를 추출합니다.
    points = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value:
                points.append((r, c))

    # 'True'인 점이 없으면 빈 리스트 반환
    if not points:
        return []

    # 행(row)을 기준으로, 행이 같으면 열(col)을 기준으로 정렬합니다.
    # 이 정렬은 DP를 효율적으로 계산하기 위해 필수적입니다.
    points.sort()

    n = len(points)
    dp = [1] * n  # dp[i]는 points[i]에서 끝나는 최장 시퀀스의 길이를 저장
    parent = [-1] * n  # 경로 복원을 위해 이전 요소의 인덱스를 저장

    # 2. 동적 계획법을 사용하여 최장 길이를 계산합니다.
    for i in range(n):
        for j in range(i):
            # 증가 조건 확인: row와 col이 모두 증가해야 함
            if points[j][0] < points[i][0] and points[j][1] < points[i][1]:
                # 더 긴 시퀀스를 만들 수 있다면 dp와 parent 값을 갱신
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

    # 3. 최장 시퀀스의 마지막 요소 인덱스를 찾습니다.
    if not dp:
        return []

    max_len = 0
    end_index = -1
    for i in range(n):
        if dp[i] > max_len:
            max_len = dp[i]
            end_index = i

    # 4. parent 배열을 역추적하여 경로를 복원합니다.
    sequence = []
    current_index = end_index
    while current_index != -1:
        sequence.append(points[current_index])
        current_index = parent[current_index]

    # 시퀀스가 시작점부터 나오도록 순서를 뒤집어줍니다.
    return sequence[::-1]


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    m, n = 192, 224
    S = (0.2 * torch.randn(m, n, device=device)).clamp_min(-6.0)  # base log-scores
    gDo = torch.full((m,), -1.1, device=device)
    gDe = torch.full((m,), -0.3, device=device)
    gIo = torch.full((n,), -1.1, device=device)
    gIe = torch.full((n,), -0.3, device=device)

    # batched μ (3 values at once)
    mu = torch.tensor([-0.2, 0.0, 0.2], device=device)

    print("Running")
    out_global = forward_backward(S, gDo, gDe, gIo, gIe, mu, band=(0, n - 1), block_size=128)
    print("logZ (global):", out_global["logZ"])

    # Check posterior probabilities
    print("\nChecking posterior probabilities...")
    # Use the logZ from the corresponding 'local' setting for posterior calculation
    log_P_M_global = out_global["F_M"] + out_global["B_M"] - out_global["logZ"].view(-1, 1, 1)
    P_M_global = torch.exp(log_P_M_global)
    print("Max P_M (global logZ):", P_M_global.max().item())
    print("Min P_M (global logZ):", P_M_global.min().item())
