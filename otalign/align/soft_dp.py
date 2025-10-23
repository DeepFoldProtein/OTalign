# softdp_triton_full.py
# 3-state alignment CRF (M/D/I) soft-DP with anti-diagonal (wavefront) schedule.
# Both FORWARD and BACKWARD passes are implemented as exact Triton kernels.
# A convenience API returns the six matrices + logZ.
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

    # --- Open tailing gaps (free ends) & end cell ---
    # end cell: (m-1, n-1) → log(1)=0
    is_end = (i == m - 1) & (j == n - 1)
    bm_k = tl.where(is_end, 0.0, bm_k)
    bd_k = tl.where(is_end, 0.0, bd_k)
    bi_k = tl.where(is_end, 0.0, bi_k)

    # free trailing deletions along the last column (j == n-1, i < m-1)
    is_last_col = (j == n - 1) & (i < m - 1)
    bd_k = tl.where(is_last_col & valid, 0.0, bd_k)

    # free trailing insertions along the last row (i == m-1, j < n - 1)
    is_last_row = (i == m - 1) & (j < n - 1)
    bi_k = tl.where(is_last_row & valid, 0.0, bi_k)

    # --- Store results ---
    tl.store(BM_k + b * Lk + offs, tl.where(valid, bm_k, NEG))
    tl.store(BD_k + b * Lk + offs, tl.where(valid, bd_k, NEG))
    tl.store(BI_k + b * Lk + offs, tl.where(valid, bi_k, NEG))


# -------------------------- Triton drivers + API --------------------------


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


# -------------------------- Batched Triton drivers + API --------------------------


@triton.jit
def fwd_diag_kernel_batched(
    # prev-2
    FM_km2,
    FD_km2,
    FI_km2,  # (B, L2)
    # prev-1
    FM_km1,
    FD_km1,
    FI_km1,  # (B, L1)
    # params
    S_M,  # (B, m, n)
    gDo,
    gDe,  # (B, m), (B, m)
    gIo,
    gIe,  # (B, n), (B, n)
    lens1,
    lens2,  # (B,), (B,)
    # outputs
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
    pid = tl.program_id(0)
    b = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Lk

    len1 = tl.load(lens1 + b)
    len2 = tl.load(lens2 + b)

    i = i_min + offs
    j = k - i
    valid = mask & (i >= 0) & (i < len1) & (j >= 0) & (j < len2)

    # S+mu
    offset = tl.where(valid, i * n + j, 0)
    s_ij = tl.load(S_M + b * m * n + offset, mask=valid, other=NEG)

    # M from k-2
    tM = (i - 1) - i2_min
    has_km2 = L2 > 0
    maskM = valid & has_km2 & (tM >= 0) & (tM < L2)
    base2 = b * L2
    tM_safe = tl.where(maskM, tM, 0)
    fm2 = tl.load(FM_km2 + base2 + tM_safe, mask=maskM, other=NEG)
    fd2 = tl.load(FD_km2 + base2 + tM_safe, mask=maskM, other=NEG)
    fi2 = tl.load(FI_km2 + base2 + tM_safe, mask=maskM, other=NEG)
    mx2 = tl.maximum(fm2, tl.maximum(fd2, fi2))
    lse2 = tl.where(mx2 > NEG, mx2 + tl.log(tl.exp(fm2 - mx2) + tl.exp(fd2 - mx2) + tl.exp(fi2 - mx2)), NEG)

    # If no parent (not maskM), but if start cell, set to s_ij, else NEG
    is_boundary_start = ((i == 0) | (j == 0)) & ~maskM
    FMv = tl.where(maskM & (lse2 > NEG), s_ij + lse2, tl.where(is_boundary_start, s_ij, NEG))

    # D from k-1
    tD = (i - 1) - i1_min
    has_km1 = L1 > 0
    maskD = valid & has_km1 & (tD >= 0) & (tD < L1)
    base1 = b * L1
    tD_safe = tl.where(maskD, tD, 0)
    fm1D = tl.load(FM_km1 + base1 + tD_safe, mask=maskD, other=NEG)
    fd1D = tl.load(FD_km1 + base1 + tD_safe, mask=maskD, other=NEG)
    i_safe = tl.where(maskD, i, 0)
    gDo_i = tl.load(gDo + b * m + i_safe, mask=maskD, other=0.0)
    gDe_i = tl.load(gDe + b * m + i_safe, mask=maskD, other=0.0)
    cD1 = fm1D + gDo_i
    cD2 = fd1D + gDe_i
    mxD = tl.maximum(cD1, cD2)
    lseD = tl.where(mxD > NEG, mxD + tl.log(tl.exp(cD1 - mxD) + tl.exp(cD2 - mxD)), NEG)
    FDv = tl.where(maskD, lseD, NEG)

    # I from k-1
    tI = i - i1_min
    maskI = valid & has_km1 & (tI >= 0) & (tI < L1)
    tI_safe = tl.where(maskI, tI, 0)
    fm1I = tl.load(FM_km1 + base1 + tI_safe, mask=maskI, other=NEG)
    fi1I = tl.load(FI_km1 + base1 + tI_safe, mask=maskI, other=NEG)
    j_safe = tl.where(maskI, j, 0)
    gIo_j = tl.load(gIo + b * n + j_safe, mask=maskI, other=0.0)
    gIe_j = tl.load(gIe + b * n + j_safe, mask=maskI, other=0.0)
    cI1 = fm1I + gIo_j
    cI2 = fi1I + gIe_j
    mxI = tl.maximum(cI1, cI2)
    lseI = tl.where(mxI > NEG, mxI + tl.log(tl.exp(cI1 - mxI) + tl.exp(cI2 - mxI)), NEG)
    FIv = tl.where(maskI, lseI, NEG)

    # Overrides for free leading gaps
    is_j0 = (j == 0) & (i > 0)
    FDv = tl.where(is_j0 & valid, 0.0, FDv)
    is_i0 = (i == 0) & (j > 0)
    FIv = tl.where(is_i0 & valid, 0.0, FIv)

    # store
    tl.store(FM_k + b * Lk + offs, tl.where(mask, FMv, NEG), mask=mask)
    tl.store(FD_k + b * Lk + offs, tl.where(mask, FDv, NEG), mask=mask)
    tl.store(FI_k + b * Lk + offs, tl.where(mask, FIv, NEG), mask=mask)


@triton.jit
def bwd_diag_kernel_batched(
    # successors k+2
    BM_kp2,
    BD_kp2,
    BI_kp2,  # (B, L2)
    # k+1
    BM_kp1,
    BD_kp1,
    BI_kp1,  # (B, L1)
    # params
    S_M,  # (B, m, n)
    gDo,
    gDe,  # (B, m), (B, m)
    gIo,
    gIe,  # (B, n), (B, n)
    lens1,
    lens2,  # (B,), (B,)
    # outputs
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

    len1 = tl.load(lens1 + b)
    len2 = tl.load(lens2 + b)

    i = i_min + offs
    j = k - i
    valid = mask & (i >= 0) & (i < len1) & (j >= 0) & (j < len2)

    ip1 = i + 1
    jp1 = j + 1
    safe_ip1 = tl.where(ip1 < len1, ip1, 0)
    safe_jp1 = tl.where(jp1 < len2, jp1, 0)

    # Successor (i+1, j+1) on k+2 for M successor
    tM_succ = ip1 - i2_min
    has_kp2 = L2 > 0
    mask_M_succ = valid & has_kp2 & (tM_succ >= 0) & (tM_succ < L2) & (ip1 < len1) & (jp1 < len2)
    succ_offset = tl.where(mask_M_succ, safe_ip1 * n + safe_jp1, 0)
    s_m_succ = tl.load(S_M + b * m * n + succ_offset, mask=mask_M_succ, other=NEG)
    tM_safe = tl.where(mask_M_succ, tM_succ, 0)
    bm_succ = tl.load(BM_kp2 + b * L2 + tM_safe, mask=mask_M_succ, other=NEG)

    # Successor (i+1, j) on k+1 for D
    tD_succ = ip1 - i1_min
    has_kp1 = L1 > 0
    mask_D_succ = valid & has_kp1 & (tD_succ >= 0) & (tD_succ < L1) & (ip1 < len1)
    tD_safe = tl.where(mask_D_succ, tD_succ, 0)
    bd_succ = tl.load(BD_kp1 + b * L1 + tD_safe, mask=mask_D_succ, other=NEG)
    ip1_safe = tl.where(mask_D_succ, safe_ip1, 0)
    gDo_ip1 = tl.load(gDo + b * m + ip1_safe, mask=mask_D_succ, other=0.0)
    gDe_ip1 = tl.load(gDe + b * m + ip1_safe, mask=mask_D_succ, other=0.0)

    # Successor (i, j+1) on k+1 for I
    tI_succ = i - i1_min
    mask_I_succ = valid & has_kp1 & (tI_succ >= 0) & (tI_succ < L1) & (jp1 < len2)
    tI_safe = tl.where(mask_I_succ, tI_succ, 0)
    bi_succ = tl.load(BI_kp1 + b * L1 + tI_safe, mask=mask_I_succ, other=NEG)
    jp1_safe = tl.where(mask_I_succ, safe_jp1, 0)
    gIo_jp1 = tl.load(gIo + b * n + jp1_safe, mask=mask_I_succ, other=0.0)
    gIe_jp1 = tl.load(gIe + b * n + jp1_safe, mask=mask_I_succ, other=0.0)

    # Compute B_M
    termM1 = tl.where(mask_M_succ, s_m_succ + bm_succ, NEG)
    termM2 = tl.where(mask_D_succ, gDo_ip1 + bd_succ, NEG)
    termM3 = tl.where(mask_I_succ, gIo_jp1 + bi_succ, NEG)
    mxM = tl.maximum(termM1, tl.maximum(termM2, termM3))
    has_M = (termM1 > NEG) | (termM2 > NEG) | (termM3 > NEG)
    expM1 = tl.exp(termM1 - mxM)
    expM2 = tl.exp(termM2 - mxM)
    expM3 = tl.exp(termM3 - mxM)
    sum_expM = expM1 + expM2 + expM3
    lseM = mxM + tl.log(sum_expM)
    BMv = tl.where(has_M, lseM, NEG)

    # Compute B_D
    termD1 = tl.where(mask_M_succ, s_m_succ + bm_succ, NEG)
    termD2 = tl.where(mask_D_succ, gDe_ip1 + bd_succ, NEG)
    mxD = tl.maximum(termD1, termD2)
    has_D = (termD1 > NEG) | (termD2 > NEG)
    expD1 = tl.exp(termD1 - mxD)
    expD2 = tl.exp(termD2 - mxD)
    sum_expD = expD1 + expD2
    lseD = mxD + tl.log(sum_expD)
    BDv = tl.where(has_D, lseD, NEG)

    # Compute B_I
    termI1 = tl.where(mask_M_succ, s_m_succ + bm_succ, NEG)
    termI2 = tl.where(mask_I_succ, gIe_jp1 + bi_succ, NEG)
    mxI = tl.maximum(termI1, termI2)
    has_I = (termI1 > NEG) | (termI2 > NEG)
    expI1 = tl.exp(termI1 - mxI)
    expI2 = tl.exp(termI2 - mxI)
    sum_expI = expI1 + expI2
    lseI = mxI + tl.log(sum_expI)
    BIv = tl.where(has_I, lseI, NEG)

    # Fix: override to 0 at end cell
    is_end = (i == len1 - 1) & (j == len2 - 1)
    BMv = tl.where(is_end, 0.0, BMv)
    BDv = tl.where(is_end, 0.0, BDv)
    BIv = tl.where(is_end, 0.0, BIv)

    # Overrides for free trailing gaps
    is_last_col = (j == len2 - 1) & (i < len1 - 1)
    BDv = tl.where(is_last_col & valid, 0.0, BDv)
    is_last_row = (i == len1 - 1) & (j < len2 - 1)
    BIv = tl.where(is_last_row & valid, 0.0, BIv)

    # store
    tl.store(BM_k + b * Lk + offs, tl.where(valid, BMv, NEG), mask=mask)
    tl.store(BD_k + b * Lk + offs, tl.where(valid, BDv, NEG), mask=mask)
    tl.store(BI_k + b * Lk + offs, tl.where(valid, BIv, NEG), mask=mask)


@torch.no_grad()
def softdp_forward_triton_batched(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    mu: torch.Tensor,
    lens1: torch.Tensor,
    lens2: torch.Tensor,
    block_size: int = 128,
):
    device, dtype = S.device, S.dtype
    NEG = _neg_inf(dtype)
    B, m, n = S.shape
    K = m + n - 1

    S_M = S + mu[:, None, None]

    FM = torch.full((B, m, n), NEG, device=device, dtype=dtype)
    FD = torch.full_like(FM, NEG)
    FI = torch.full_like(FM, NEG)

    KPm2 = None
    KPm1 = None

    for k in range(K):
        i_min, _, Lk = _anti_diag_bounds(k, m, n, 0, n - 1)
        if Lk == 0:
            continue

        def bounds(k_):
            if k_ < 0:
                return None
            imn, _, L = _anti_diag_bounds(k_, m, n, 0, n - 1)
            return (imn, L) if L > 0 else None

        b_m1 = bounds(k - 1)
        b_m2 = bounds(k - 2)

        FMk, FDk, FIk = (torch.full((B, Lk), NEG, device=device, dtype=dtype) for _ in range(3))

        i1_min = b_m1[0] if b_m1 else 0
        L1 = b_m1[1] if b_m1 else 0
        if KPm1:
            FM1, FD1, FI1 = KPm1
        else:
            FM1 = torch.empty((B, 1), device=device, dtype=dtype)
            FD1 = torch.empty((B, 1), device=device, dtype=dtype)
            FI1 = torch.empty((B, 1), device=device, dtype=dtype)

        i2_min = b_m2[0] if b_m2 else 0
        L2 = b_m2[1] if b_m2 else 0
        if KPm2:
            FM2, FD2, FI2 = KPm2
        else:
            FM2 = torch.empty((B, 1), device=device, dtype=dtype)
            FD2 = torch.empty((B, 1), device=device, dtype=dtype)
            FI2 = torch.empty((B, 1), device=device, dtype=dtype)

        grid = ((Lk + block_size - 1) // block_size, B)
        fwd_diag_kernel_batched[grid](
            FM2, FD2, FI2, FM1, FD1, FI1, S_M, gD_open, gD_ext, gI_open, gI_ext, lens1, lens2, FMk, FDk, FIk, i_min, k, Lk, i1_min, L1, i2_min, L2, m, n, NEG, BLOCK=block_size
        )

        t = torch.arange(Lk, device=device)
        i, j = i_min + t, k - (i_min + t)
        FM[:, i, j] = FMk
        FD[:, i, j] = FDk
        FI[:, i, j] = FIk

        KPm2 = KPm1
        KPm1 = (FMk, FDk, FIk)

    # Fixed semi-global logZ: LSE over all states in last row and last col, with double-count correction for corner
    b_idx = torch.arange(B, device=device)
    i_last = lens1 - 1
    j_last = lens2 - 1

    J = torch.arange(n, device=device)[None, :]
    mask_j = J < lens2[:, None]
    F_M_last_row = FM[b_idx[:, None], i_last[:, None], J.expand(B, n)]
    F_D_last_row = FD[b_idx[:, None], i_last[:, None], J.expand(B, n)]
    F_I_last_row = FI[b_idx[:, None], i_last[:, None], J.expand(B, n)]
    F_last_row = torch.stack([F_M_last_row, F_D_last_row, F_I_last_row], dim=0)  # (3, B, n)
    F_last_row = F_last_row.masked_fill(~mask_j.unsqueeze(0), NEG)

    I = torch.arange(m, device=device)[None, :]
    mask_i = I < lens1[:, None]
    F_M_last_col = FM[b_idx[:, None], I.expand(B, m), j_last[:, None]]
    F_D_last_col = FD[b_idx[:, None], I.expand(B, m), j_last[:, None]]
    F_I_last_col = FI[b_idx[:, None], I.expand(B, m), j_last[:, None]]
    F_last_col = torch.stack([F_M_last_col, F_D_last_col, F_I_last_col], dim=0)  # (3, B, m)
    F_last_col = F_last_col.masked_fill(~mask_i.unsqueeze(0), NEG)

    # Corner (m-1, n-1): allow M, D, I
    F_M_end = FM[b_idx, i_last, j_last]
    F_D_end = FD[b_idx, i_last, j_last]
    F_I_end = FI[b_idx, i_last, j_last]
    z_corner = torch.logsumexp(torch.stack([F_M_end, F_D_end, F_I_end], dim=0), dim=0)  # (B,)

    # Last row (i == m-1, j = 0..n-2): allow M and I only
    J_no_corner = torch.arange(n - 1, device=device)[None, :]
    mask_j_no_corner = J_no_corner < (lens2[:, None] - 1).clamp_min(0)

    F_M_last_row_nc = FM[b_idx[:, None], i_last[:, None], J_no_corner.expand(B, n - 1)]
    F_I_last_row_nc = FI[b_idx[:, None], i_last[:, None], J_no_corner.expand(B, n - 1)]

    F_last_row_stack = torch.stack([F_M_last_row_nc, F_I_last_row_nc], dim=0)  # (2, B, n-1)
    F_last_row_stack = F_last_row_stack.masked_fill(~mask_j_no_corner.unsqueeze(0), NEG)
    z_row_without = torch.logsumexp(F_last_row_stack, dim=0).logsumexp(dim=1)

    # Last column (j == n-1, i = 0..m-2): allow M and D only
    I_no_corner = torch.arange(m - 1, device=device)[None, :]
    mask_i_no_corner = I_no_corner < (lens1[:, None] - 1).clamp_min(0)

    F_M_last_col_nc = FM[b_idx[:, None], I_no_corner.expand(B, m - 1), j_last[:, None]]
    F_D_last_col_nc = FD[b_idx[:, None], I_no_corner.expand(B, m - 1), j_last[:, None]]

    F_last_col_stack = torch.stack([F_M_last_col_nc, F_D_last_col_nc], dim=0)  # (2, B, m-1)
    F_last_col_stack = F_last_col_stack.masked_fill(~mask_i_no_corner.unsqueeze(0), NEG)
    z_col_without = torch.logsumexp(F_last_col_stack, dim=0).logsumexp(dim=1)  # (B,)

    # Final logZ
    logZ = torch.logsumexp(torch.stack([z_row_without, z_col_without, z_corner], dim=0), dim=0)

    return FM, FD, FI, logZ


@torch.no_grad()
def softdp_backward_triton_batched(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    mu: torch.Tensor,
    lens1: torch.Tensor,
    lens2: torch.Tensor,
    block_size: int = 128,
):
    device, dtype = S.device, S.dtype
    NEG = _neg_inf(dtype)
    B, m, n = S.shape
    K = m + n - 1

    S_M = S + mu[:, None, None]

    BM = torch.full((B, m, n), NEG, device=device, dtype=dtype)
    BD = torch.full_like(BM, NEG)
    BI = torch.full_like(BM, NEG)

    KP2 = KP1 = None

    # Removed initial set to 0 at ends, let kernel handle

    for k in range(K - 1, -1, -1):  # Fix: start from K-1 to include potential end diagonals
        i_min, _, Lk = _anti_diag_bounds(k, m, n, 0, n - 1)
        if Lk == 0:
            KP2, KP1 = KP1, None
            continue

        def bounds(k_):
            if k_ < 0 or k_ >= K:
                return None
            imn, _, L = _anti_diag_bounds(k_, m, n, 0, n - 1)
            return (imn, L) if L > 0 else None

        b1 = bounds(k + 1)
        b2 = bounds(k + 2)

        BMk, BDk, BIk = (torch.full((B, Lk), NEG, device=device, dtype=dtype) for _ in range(3))

        i1_min, L1 = (b1[0], b1[1]) if b1 else (0, 0)
        BM1, BD1, BI1 = KP1 if KP1 else (torch.empty((B, 1), device=device, dtype=dtype), torch.empty((B, 1), device=device, dtype=dtype), torch.empty((B, 1), device=device, dtype=dtype))
        i2_min, L2 = (b2[0], b2[1]) if b2 else (0, 0)
        BM2, BD2, BI2 = KP2 if KP2 else (torch.empty((B, 1), device=device, dtype=dtype), torch.empty((B, 1), device=device, dtype=dtype), torch.empty((B, 1), device=device, dtype=dtype))

        grid = ((Lk + block_size - 1) // block_size, B)
        bwd_diag_kernel_batched[grid](
            BM2, BD2, BI2, BM1, BD1, BI1, S_M, gD_open, gD_ext, gI_open, gI_ext, lens1, lens2, BMk, BDk, BIk, i_min, k, Lk, i1_min, L1, i2_min, L2, m, n, NEG, BLOCK=block_size
        )

        t = torch.arange(Lk, device=device)
        i, j = i_min + t, k - (i_min + t)
        BM[:, i, j] = BMk
        BD[:, i, j] = BDk
        BI[:, i, j] = BIk

        KP2, KP1 = KP1, (BMk, BDk, BIk)

    return BM, BD, BI


@torch.no_grad()
def forward_backward_batched(
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    mu: torch.Tensor,
    lens1: torch.Tensor,
    lens2: torch.Tensor,
    block_size: int = 128,
):
    """Batched version of forward_backward."""
    assert S.ndim == 3, "S must be (B, m, n)"
    B, m, n = S.shape
    assert gD_open.shape == (B, m) and gD_ext.shape == (B, m)
    assert gI_open.shape == (B, n) and gI_ext.shape == (B, n)
    assert mu.shape == (B,)
    assert lens1.shape == (B,) and lens2.shape == (B,)

    FM, FD, FI, logZ = softdp_forward_triton_batched(S, gD_open, gD_ext, gI_open, gI_ext, mu, lens1, lens2, block_size=block_size)
    BM, BD, BI = softdp_backward_triton_batched(S, gD_open, gD_ext, gI_open, gI_ext, mu, lens1, lens2, block_size=block_size)
    return {"F_M": FM, "F_D": FD, "F_I": FI, "B_M": BM, "B_D": BD, "B_I": BI, "logZ": logZ}


@torch.no_grad()
def compute_logZ_and_der1_batched(
    S: torch.Tensor,  # (B, m, n)
    gD_open: torch.Tensor,  # (B, m)
    gD_ext: torch.Tensor,  # (B, m)
    gI_open: torch.Tensor,  # (B, n)
    gI_ext: torch.Tensor,  # (B, n)
    mu: torch.Tensor,  # (B_mu,)
    lens1: torch.Tensor,  # (B,)
    lens2: torch.Tensor,  # (B,)
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    # If multiple mu, repeat inputs across mu (assuming each sequence gets all mu)
    # if B_mu > B:
    #     repeats = B_mu // B
    #     if B_mu % B != 0:
    #         raise AssertionError("B_mu must be multiple of B for repeating stats across mu")
    #     S = S.repeat_interleave(repeats, dim=0)
    #     gD_open = gD_open.repeat_interleave(repeats, dim=0)
    #     gD_ext = gD_ext.repeat_interleave(repeats, dim=0)
    #     gI_open = gI_open.repeat_interleave(repeats, dim=0)
    #     gI_ext = gI_ext.repeat_interleave(repeats, dim=0)
    #     lens1 = lens1.repeat_interleave(repeats)
    #     lens2 = lens2.repeat_interleave(repeats)
    # elif B_mu < B:
    #     raise ValueError("B_mu cannot be less than B")

    dp = forward_backward_batched(S, gD_open, gD_ext, gI_open, gI_ext, mu, lens1, lens2, block_size=block_size)
    logZ = dp["logZ"]

    # der1 = sum_{i,j} P_M[i,j] = expected #matches
    log_P_M = dp["F_M"] + dp["B_M"] - logZ[:, None, None]
    P_M = torch.exp(log_P_M)

    # Mask out padded regions before summing to get der1
    B_eff, m, n = P_M.shape
    i_coords = torch.arange(m, device=P_M.device).view(1, m, 1)
    j_coords = torch.arange(n, device=P_M.device).view(1, 1, n)
    mask = (i_coords < lens1.view(B_eff, 1, 1)) & (j_coords < lens2.view(B_eff, 1, 1))
    P_M.masked_fill_(~mask, 0.0)
    der1 = P_M.sum(dim=(-1, -2))  # Batched sum

    return logZ, der1


@torch.no_grad()
def der1_bisect(
    value: torch.Tensor,  # (B,)
    S: torch.Tensor,
    gD_open: torch.Tensor,
    gD_ext: torch.Tensor,
    gI_open: torch.Tensor,
    gI_ext: torch.Tensor,
    lens1: torch.Tensor,  # (B,)
    lens2: torch.Tensor,  # (B,)
    initial_low: float = 0.0,
    initial_high: float = 50.0,
    max_iters: int = 50,
    tol: float = 1e-3,
    block_size: int = 128,
) -> torch.Tensor:
    B = S.shape[0]
    device = S.device

    a = torch.full((B,), initial_low, device=device)
    b = torch.full((B,), initial_high, device=device)

    def f(mu: torch.Tensor) -> torch.Tensor:
        _, der = compute_logZ_and_der1_batched(S, gD_open, gD_ext, gI_open, gI_ext, mu, lens1, lens2, block_size=block_size)
        return der - value

    fa = f(a)
    # fb = f(b)

    # if torch.any(fa * fb >= 0):
    #     print("Warning: Root is not bracketed or function has the same sign at endpoints.")

    for _ in range(max_iters):
        c = 0.5 * (a + b)
        fc = f(c)

        if torch.all(torch.abs(fc) < tol) or torch.all((b - a) * 0.5 < tol):
            return c

        mask = fa * fc < 0
        b = torch.where(mask, c, b)
        a = torch.where(~mask, c, a)

        fa = torch.where(~mask, fc, fa)

    return 0.5 * (a + b)


# Example usage
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    B = 4  # Set B=1 for multiple mu example to avoid error
    m, n = 100, 120

    S = 0.2 * torch.randn(B, m, n, device=device)
    gDo = torch.full((B, m), -1.2, device=device)
    gDe = torch.full((B, m), -0.3, device=device)
    gIo = torch.full((B, n), -1.2, device=device)
    gIe = torch.full((B, n), -0.3, device=device)
    lens1 = torch.full((B,), m, device=device)
    lens2 = torch.full((B,), n, device=device)

    mu = S.new_full((B,), 3.0)
    logZ, der1 = compute_logZ_and_der1_batched(S, gDo, gDe, gIo, gIe, mu, lens1, lens2)
    print("logZ:", logZ)
    print("der1:", der1)
