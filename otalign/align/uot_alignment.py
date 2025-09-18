from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


try:
    from numba import njit

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# ---------- fast helpers ----------


def _sigmoid_neg_kx(x: np.ndarray, k: float) -> np.ndarray:
    # sigmoid(-k*x) = 1 / (1 + exp(k*x)); stable for moderate |k*x|
    return 1.0 / (1.0 + np.exp(k * x))


def compute_match_scores_from_transport(P: np.ndarray, eps: float = 1e-12, scale: float = 1.0) -> np.ndarray:
    """
    Compute per-cell PMI-like scores, with fewer temporaries and in-place ops.
    """
    P = np.asarray(P, dtype=np.float64, order="C")
    total = P.sum() + eps

    # pij = P / total (reuse buffer)
    pij = P.astype(np.float64, copy=True)
    pij /= total

    # row/col marginals
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)

    # S = log p(i,j) - log p(i) - log p(j)
    # use out= to avoid temporaries and reuse pij buffer for logs
    np.log(pij + eps, out=pij)
    np.log(pi + eps, out=pi)
    np.log(pj + eps, out=pj)
    pij -= pi
    pij -= pj  # pij now holds S

    if scale != 1.0:
        pij *= scale
    return pij  # (Lq, Lt)


def compute_gap_penalties_from_uot(
    P: np.ndarray,
    f: np.ndarray | None = None,
    g: np.ndarray | None = None,
    go_base: float = 8.0,
    ge_base: float = 1.0,
    gamma: float = 1.0,
    eta: float = 0.25,
    clip_upper: float = 4.0,
    clip_lower: float = 0.25,
    k_f: float = 0.75,
    k_g: float = 0.75,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=np.float64, order="C")
    Lq, Lt = P.shape

    row_mass = P.sum(axis=1)
    col_mass = P.sum(axis=0)

    # robust medians on non-zeros
    med_r = np.median(row_mass[row_mass > eps]) if np.any(row_mass > eps) else 1.0
    med_c = np.median(col_mass[col_mass > eps]) if np.any(col_mass > eps) else 1.0

    rtilde = np.clip(row_mass / (med_r + eps), clip_lower, clip_upper)
    ctilde = np.clip(col_mass / (med_c + eps), clip_lower, clip_upper)

    if gamma != 1.0:
        rpow = rtilde**gamma
        cpow = ctilde**gamma
    else:
        rpow = rtilde
        cpow = ctilde

    if f is not None:
        f = np.asarray(f, dtype=np.float64).reshape(-1)
        f_scaled = (f - np.median(f)) / (np.std(f) + eps)
        factor_q = rpow * _sigmoid_neg_kx(f_scaled, k_f)
    else:
        factor_q = rpow

    if g is not None:
        g = np.asarray(g, dtype=np.float64).reshape(-1)
        g_scaled = (g - np.median(g)) / (np.std(g) + eps)
        factor_t = cpow * _sigmoid_neg_kx(g_scaled, k_g)
    else:
        factor_t = cpow

    go_q = np.maximum(ge_base, go_base * factor_q)
    ge_q = np.maximum(eta * ge_base, ge_base * factor_q)
    go_t = np.maximum(ge_base, go_base * factor_t)
    ge_t = np.maximum(eta * ge_base, ge_base * factor_t)

    # add dummy at index 0 for 1-based DP indexing
    go_q = np.concatenate(([go_q[0]], go_q))
    ge_q = np.concatenate(([ge_q[0]], ge_q))
    go_t = np.concatenate(([go_t[0]], go_t))
    ge_t = np.concatenate(([ge_t[0]], ge_t))

    return go_q, ge_q, go_t, ge_t


def _to_cigar(ops: List[str]) -> str:
    if not ops:
        return ""
    out = []
    count = 1
    prev = ops[0]
    for x in ops[1:]:
        if x == prev:
            count += 1
        else:
            out.append(f"{count}{prev}")
            prev = x
            count = 1
    out.append(f"{count}{prev}")
    return "".join(out)


@njit(cache=True, fastmath=True)
def _dp_core_numba(S, go_q, ge_q, go_t, ge_t, mode, band):
    Lq, Lt = S.shape
    neg_inf = -1e18

    M = np.full((Lq + 1, Lt + 1), neg_inf)
    X = np.full((Lq + 1, Lt + 1), neg_inf)
    Y = np.full((Lq + 1, Lt + 1), neg_inf)
    ptrM = np.zeros((Lq + 1, Lt + 1), np.int8)
    ptrX = np.zeros((Lq + 1, Lt + 1), np.int8)
    ptrY = np.zeros((Lq + 1, Lt + 1), np.int8)

    # Initialization for local alignment
    if mode == 4:
        # For local alignment, the score can start anywhere, so the first row/col are 0.
        for i in range(Lq + 1):
            M[i, 0] = 0.0
        for j in range(Lt + 1):
            M[0, j] = 0.0
    else:
        M[0, 0] = 0.0

    # Initialization step: Defines how starting gaps are penalized.
    if mode == 0:  # global alignment
        # Penalize gaps at the beginning of both query and template.
        for i in range(1, Lq + 1):
            open_score = M[i - 1, 0] - go_q[i]
            extend_score = X[i - 1, 0] - ge_q[i]
            if open_score >= extend_score:
                X[i, 0] = open_score
                ptrX[i, 0] = 0
            else:
                X[i, 0] = extend_score
                ptrX[i, 0] = 1

        for j in range(1, Lt + 1):
            open_score = M[0, j - 1] - go_t[j]
            extend_score = Y[0, j - 1] - ge_t[j]
            if open_score >= extend_score:
                Y[0, j] = open_score
                ptrY[0, j] = 0
            else:
                Y[0, j] = extend_score
                ptrY[0, j] = 1
    elif mode == 1:  # query-open glocal (free starting gaps for template)
        # Penalize gaps at the start of the query (first column).
        for i in range(1, Lq + 1):
            open_score = M[i - 1, 0] - go_q[i]
            extend_score = X[i - 1, 0] - ge_q[i]
            if open_score >= extend_score:
                X[i, 0] = open_score
                ptrX[i, 0] = 0
            else:
                X[i, 0] = extend_score
                ptrX[i, 0] = 1
        # No penalty for starting gaps in the template (first row).
        for j in range(1, Lt + 1):
            M[0, j] = 0.0

    elif mode == 2:  # template-open glocal (free starting gaps for query)
        # No penalty for starting gaps in the query (first column).
        for i in range(1, Lq + 1):
            M[i, 0] = 0.0
        # Penalize gaps at the start of the template (first row).
        for j in range(1, Lt + 1):
            open_score = M[0, j - 1] - go_t[j]
            extend_score = Y[0, j - 1] - ge_t[j]
            if open_score >= extend_score:
                Y[0, j] = open_score
                ptrY[0, j] = 0
            else:
                Y[0, j] = extend_score
                ptrY[0, j] = 1

    elif mode == 3:  # both-open glocal (free starting gaps for both)
        # No penalty for starting gaps in either sequence.
        for i in range(1, Lq + 1):
            M[i, 0] = 0.0
        for j in range(1, Lt + 1):
            M[0, j] = 0.0

    # mode 5 (local) initialization is handled before the main block.

    # Main recursion (fill matrices)
    for i in range(1, Lq + 1):
        j_start = 1
        j_end = Lt
        if band > 0:
            if i - band > j_start:
                j_start = i - band
            if i + band < j_end:
                j_end = i + band

        for j in range(j_start, j_end + 1):
            sij = S[i - 1, j - 1]
            m0 = M[i - 1, j - 1]
            m1 = X[i - 1, j - 1]
            m2 = Y[i - 1, j - 1]
            # argmax over 3
            if m0 >= m1 and m0 >= m2:
                prev = 0
                prev_val = m0
            elif m1 >= m2:
                prev = 1
                prev_val = m1
            else:
                prev = 2
                prev_val = m2

            m_val = sij + prev_val

            # Key difference for local alignment: score cannot be negative.
            if mode == 4:
                if m_val < 0:
                    m_val = 0

            M[i, j] = m_val
            ptrM[i, j] = prev

            # X: gap in target (insert in query)
            open_score = M[i - 1, j] - go_q[i]
            extend_score = X[i - 1, j] - ge_q[i]
            if open_score >= extend_score:
                X[i, j] = open_score
                ptrX[i, j] = 0
            else:
                X[i, j] = extend_score
                ptrX[i, j] = 1

            # Y: gap in query (delete in query)
            open_score = M[i, j - 1] - go_t[j]
            extend_score = Y[i, j - 1] - ge_t[j]
            if open_score >= extend_score:
                Y[i, j] = open_score
                ptrY[i, j] = 0
            else:
                Y[i, j] = extend_score
                ptrY[i, j] = 1

    # Termination step: Finds the best score and the starting point for traceback.
    if mode == 0:  # global
        # Score must be at the bottom-right corner.
        end0 = M[Lq, Lt]
        end1 = X[Lq, Lt]
        end2 = Y[Lq, Lt]
        best_score = end0
        state = 0
        if end1 > best_score:
            best_score = end1
            state = 1
        if end2 > best_score:
            best_score = end2
            state = 2
        i, j = Lq, Lt
    else:
        # For glocal/local modes, traceback starts from the M matrix.
        state = 0
        if mode == 1:  # query-open: find max score in the last row
            # The entire query must align, but the template can end with a gap.
            best_score = neg_inf
            j_max = 0
            for k in range(Lt + 1):
                if M[Lq, k] > best_score:
                    best_score = M[Lq, k]
                    j_max = k
            i, j = Lq, j_max
        elif mode == 2:  # template-open: find max score in the last column
            # The entire template must align, but the query can end with a gap.
            best_score = neg_inf
            i_max = 0
            for k in range(Lq + 1):
                if M[k, Lt] > best_score:
                    best_score = M[k, Lt]
                    i_max = k
            i, j = i_max, Lt
        elif mode == 3 or mode == 4:  # both-open or local: find max score in the entire matrix
            # Alignment can start and end anywhere (local-like behavior).
            best_score = neg_inf
            i_max, j_max = 0, 0
            for r in range(Lq + 1):
                for c in range(Lt + 1):
                    if M[r, c] > best_score:
                        best_score = M[r, c]
                        i_max, j_max = r, c
            i, j = i_max, j_max

    return M, X, Y, ptrM, ptrX, ptrY, i, j, state, best_score


def _dp_core_numpy(S, go_q, ge_q, go_t, ge_t, mode, band):
    # Identical math to the numba core but in pure Python/NumPy loops.
    Lq, Lt = S.shape
    neg_inf = -1e18

    M = np.full((Lq + 1, Lt + 1), neg_inf, dtype=np.float64)
    X = np.full((Lq + 1, Lt + 1), neg_inf, dtype=np.float64)
    Y = np.full((Lq + 1, Lt + 1), neg_inf, dtype=np.float64)
    ptrM = np.zeros((Lq + 1, Lt + 1), dtype=np.int8)
    ptrX = np.zeros((Lq + 1, Lt + 1), dtype=np.int8)
    ptrY = np.zeros((Lq + 1, Lt + 1), dtype=np.int8)

    # Initialization for local alignment
    if mode == 4:
        # For local alignment, the score can start anywhere, so the first row/col are 0.
        M[:, 0] = 0.0
        M[0, :] = 0.0
    else:
        M[0, 0] = 0.0

    # Initialization step: Defines how starting gaps are penalized.
    if mode == 0:  # global alignment
        # Penalize gaps at the beginning of both query and template.
        for i in range(1, Lq + 1):
            open_score = M[i - 1, 0] - go_q[i]
            extend_score = X[i - 1, 0] - ge_q[i]
            if open_score >= extend_score:
                X[i, 0] = open_score
                ptrX[i, 0] = 0
            else:
                X[i, 0] = extend_score
                ptrX[i, 0] = 1
        for j in range(1, Lt + 1):
            open_score = M[0, j - 1] - go_t[j]
            extend_score = Y[0, j - 1] - ge_t[j]
            if open_score >= extend_score:
                Y[0, j] = open_score
                ptrY[0, j] = 0
            else:
                Y[0, j] = extend_score
                ptrY[0, j] = 1
    elif mode == 1:  # query-open glocal (free starting gaps for template)
        # Penalize gaps at the start of the query (first column).
        for i in range(1, Lq + 1):
            open_score = M[i - 1, 0] - go_q[i]
            extend_score = X[i - 1, 0] - ge_q[i]
            if open_score >= extend_score:
                X[i, 0] = open_score
                ptrX[i, 0] = 0
            else:
                X[i, 0] = extend_score
                ptrX[i, 0] = 1
        # No penalty for starting gaps in the template (first row).
        M[0, 1:] = 0.0

    elif mode == 2:  # template-open glocal (free starting gaps for query)
        # No penalty for starting gaps in the query (first column).
        M[1:, 0] = 0.0
        # Penalize gaps at the start of the template (first row).
        for j in range(1, Lt + 1):
            open_score = M[0, j - 1] - go_t[j]
            extend_score = Y[0, j - 1] - ge_t[j]
            if open_score >= extend_score:
                Y[0, j] = open_score
                ptrY[0, j] = 0
            else:
                Y[0, j] = extend_score
                ptrY[0, j] = 1

    elif mode == 3:  # both-open glocal (free starting gaps for both)
        # No penalty for starting gaps in either sequence.
        M[1:, 0] = 0.0
        M[0, 1:] = 0.0

    # mode 5 (local) initialization is handled before the main block.

    # Main recursion (fill matrices)
    for i in range(1, Lq + 1):
        j_start = 1
        j_end = Lt
        if band is not None and band > 0:
            j_start = max(1, i - band)
            j_end = min(Lt, i + band)

        for j in range(j_start, j_end + 1):
            sij = S[i - 1, j - 1]

            # M
            m_candidates = (M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            m_prev = 0 if (m_candidates[0] >= m_candidates[1] and m_candidates[0] >= m_candidates[2]) else (1 if m_candidates[1] >= m_candidates[2] else 2)

            m_val = sij + m_candidates[m_prev]

            # Key difference for local alignment: score cannot be negative.
            if mode == 4:
                M[i, j] = max(0.0, m_val)
            else:
                M[i, j] = m_val

            ptrM[i, j] = m_prev

            # X
            open_score = M[i - 1, j] - go_q[i]
            extend_score = X[i - 1, j] - ge_q[i]
            if open_score >= extend_score:
                X[i, j] = open_score
                ptrX[i, j] = 0
            else:
                X[i, j] = extend_score
                ptrX[i, j] = 1

            # Y
            open_score = M[i, j - 1] - go_t[j]
            extend_score = Y[i, j - 1] - ge_t[j]
            if open_score >= extend_score:
                Y[i, j] = open_score
                ptrY[i, j] = 0
            else:
                Y[i, j] = extend_score
                ptrY[i, j] = 1

    # Termination step: Finds the best score and the starting point for traceback.
    if mode == 0:  # global
        # Score must be at the bottom-right corner.
        end_candidates = (M[Lq, Lt], X[Lq, Lt], Y[Lq, Lt])
        state = int(np.argmax(end_candidates))
        i, j = Lq, Lt
        best_score = end_candidates[state]
    else:
        # For glocal/local modes, traceback starts from the M matrix.
        state = 0
        if mode == 1:  # query-open: find max score in the last row
            # The entire query must align, but the template can end with a gap.
            j = np.argmax(M[Lq, :])
            i = Lq
            best_score = M[i, j]
        elif mode == 2:  # template-open: find max score in the last column
            # The entire template must align, but the query can end with a gap.
            i = np.argmax(M[:, Lt])
            j = Lt
            best_score = M[i, j]
        elif mode == 3 or mode == 4:  # both-open or local: find max score in the entire matrix
            # Alignment can start and end anywhere (local-like behavior).
            flat_idx = np.argmax(M)
            i, j = np.unravel_index(flat_idx, M.shape)
            best_score = M[i, j]

    return M, X, Y, ptrM, ptrX, ptrY, i, j, state, best_score


def hard_alignment_from_transport(
    P: np.ndarray,
    f: np.ndarray | None = None,
    g: np.ndarray | None = None,
    mode: str = "global",
    go_base: float = 8.0,  # gap open base
    ge_base: float = 1.0,  # gap extend base
    gamma: float = 1.0,  # mass sensitivity
    eta: float = 0.25,  # extend minimum ratio
    k_f: float = 0.75,  # dual sensitivity
    k_g: float = 0.75,  # dual sensitivity
    clip_upper: float = 4.0,  # mass normalization upper limit
    clip_lower: float = 0.25,  # mass normalization lower limit
    score_scale: float = 1.0,  # alpha; score scale
    band: int | None = None,  # band width
    eps: float = 1e-12,
    mask: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute a hard alignment (CIGAR) from a UOT transport plan with optional masks.

    Mask semantics:
      - query_mask[i] == 0 (or False): query residue i+1 is *masked out*:
          it cannot match any template residue (no 'M' at that row),
          and opening/extending a gap at i+1 is made effectively free.
      - template_mask[j] == 0 (or False): template residue j+1 is masked out
          (cannot be matched); gaps at j+1 are made effectively free.
      - True/1 means normal behavior.

    Args:
        P: (Lq, Lt) transport plan.
        f, g: Dual potentials (optional).
        mode: "global", "glocal", "q2t", "t2q", or "local".
        go_base, ge_base, gamma, eta, k_f, k_g, clip_upper, clip_lower, score_scale, band, eps:
            Standard parameters as before.
        mask: Optional (Lq, Lt) boolean array. False cells are forbidden to match.

    Returns:
        Dict with keys: "cigar", "ops", "path", "score", "params".
    """
    P = np.asarray(P, dtype=np.float64, order="C")
    Lq, Lt = P.shape

    # 1) Scores from transport
    S = compute_match_scores_from_transport(P, eps=eps, scale=score_scale)

    # 2) Forbid matches on masked-out cells by setting S to -inf there
    if mask is not None:
        assert mask.shape == (Lq, Lt)
        neg_inf = -1e18
        S = np.where(mask, S, neg_inf)

    # 3) Gap penalties from UOT
    go_q, ge_q, go_t, ge_t = compute_gap_penalties_from_uot(
        P,
        f=f,
        g=g,
        go_base=go_base,
        ge_base=ge_base,
        gamma=gamma,
        eta=eta,
        k_f=k_f,
        k_g=k_g,
        clip_upper=clip_upper,
        clip_lower=clip_lower,
        eps=eps,
    )

    # 4) Mode flag and (optional) band
    _mode_flag = 0 if mode == "global" else 1 if mode == "q2t" else 2 if mode == "t2q" else 3 if mode == "glocal" else 4 if mode == "local" else 5
    if mode not in ("global", "glocal", "q2t", "t2q", "local"):
        raise ValueError(f"Invalid mode: '{mode}'")

    _band = int(band) if band is not None else 0

    # 5) Core DP (Numba or NumPy backend)
    if _HAVE_NUMBA:
        M, X, Y, ptrM, ptrX, ptrY, i, j, state, best_score = _dp_core_numba(S, go_q, ge_q, go_t, ge_t, _mode_flag, _band)
    else:
        M, X, Y, ptrM, ptrX, ptrY, i, j, state, best_score = _dp_core_numpy(S, go_q, ge_q, go_t, ge_t, _mode_flag, _band)

    # 6) Traceback
    ops = []
    path = []
    while i > 0 or j > 0:
        if state == 0:  # M
            if i == 0 or j == 0:
                break
            prev_state = int(ptrM[i, j])
            ops.append("M")
            path.append((i, j, "M"))
            i -= 1
            j -= 1
            state = prev_state
        elif state == 1:  # X => insertion (gap in target)
            if i == 0:
                break
            prev = int(ptrX[i, j])
            ops.append("I")
            path.append((i, j, "I"))
            i -= 1
            state = 0 if prev == 0 else 1
        else:  # Y => deletion (gap in query)
            if j == 0:
                break
            prev = int(ptrY[i, j])
            ops.append("D")
            path.append((i, j, "D"))
            j -= 1
            state = 0 if prev == 0 else 2

        # glocal early stop (q2t): stop at edges on open side
        if _mode_flag == 1 and (i == 0 or j == 0):
            break

    ops.reverse()
    path.reverse()
    cigar = _to_cigar(ops)

    return {
        "cigar": cigar,
        "ops": "".join(ops),
        "path": path,
        "score": float(best_score),
        "params": {
            "mode": mode,
            "go_base": go_base,
            "ge_base": ge_base,
            "gamma": gamma,
            "k_f": k_f,
            "k_g": k_g,
            "score_scale": score_scale,
            "band": band,
        },
    }


def uot_alignment_metrics_with_sinkhorn(
    a: torch.Tensor,  # [B, M]
    b: torch.Tensor,  # [B, N]
    cost_matrix: torch.Tensor,  # [B, M, N]   : C_xy = 1 - cosine
    transport_plan: torch.Tensor,  # [B, M, N]   : P_xy >= 0
    mask_a: Optional[torch.Tensor] = None,  # [B, M] bool (optional, True = valid)
    mask_b: Optional[torch.Tensor] = None,  # [B, N] bool (optional, True = valid)
    # self-terms for Sinkhorn divergence
    cost_xx: Optional[torch.Tensor] = None,  # [B, M, M]   : C_xx for (X,X)
    cost_yy: Optional[torch.Tensor] = None,  # [B, N, N]   : C_yy for (Y,Y)
    plan_xx: Optional[torch.Tensor] = None,  # [B, M, M]   : P_xx >= 0
    plan_yy: Optional[torch.Tensor] = None,  # [B, N, N]   : P_yy >= 0
    # UOT hyperparameters (same ones used to obtain the given plans)
    reg: float = 0.1,  # entropic regularization ε > 0
    lambda1: float = 1.0,  # KL penalty for source (rows)
    lambda2: float = 1.0,  # KL penalty for target (cols)
    # extras
    delta: float = 0.10,  # diagonal band width in normalized coords
    tiny: float = 1e-12,  # numerical epsilon
) -> dict[str, torch.Tensor]:
    """
    Returns a dict of batched metrics (shape [B] unless stated otherwise):
      - transport_cost, mean_cosine, mass_total, coverage_ratio
      - monotonicity_corr, mapping_tv
      - row_entropy_mean, col_entropy_mean
      - diag_band_mass, diag_band_mass_ratio
      - uot_objective_xy, uot_objective_xx, uot_objective_yy
      - sinkhorn_divergence
    """
    B, M, N = transport_plan.shape
    device = transport_plan.device
    dtype = transport_plan.dtype

    if mask_a is None:
        mask_a = torch.ones((B, M), dtype=torch.bool, device=device)
    if mask_b is None:
        mask_b = torch.ones((B, N), dtype=torch.bool, device=device)

    # ----- joint mask & masked XY plan -----
    row_mask_xy = mask_a.unsqueeze(-1)  # [B, M, 1]
    col_mask_xy = mask_b.unsqueeze(-2)  # [B, 1, N]
    joint_mask_xy = row_mask_xy & col_mask_xy  # [B, M, N]
    P_xy = transport_plan * joint_mask_xy.to(transport_plan.dtype)

    # =========================
    # Basic masses / costs (XY)
    # =========================
    mass_total = P_xy.sum(dim=(-2, -1))  # [B]
    a_mass = (a * mask_a.to(a.dtype)).sum(dim=-1)  # [B]
    b_mass = (b * mask_b.to(b.dtype)).sum(dim=-1)  # [B]
    coverage_ratio = mass_total / (torch.minimum(a_mass, b_mass) + tiny)  # [B]

    transport_cost = (P_xy * cost_matrix).sum(dim=(-2, -1))  # [B]
    mean_cosine = ((P_xy * (1.0 - cost_matrix)).sum(dim=(-2, -1))) / (mass_total + tiny)

    # ================
    # Entropies (XY)
    # ================
    row_sums_xy = P_xy.sum(dim=-1) + tiny  # [B, M]
    col_sums_xy = P_xy.sum(dim=-2) + tiny  # [B, N]

    # Row-wise normalized entropy
    q_row = P_xy / row_sums_xy.unsqueeze(-1)  # [B, M, N]
    row_entropy = -(q_row.clamp_min(tiny) * q_row.clamp_min(tiny).log()).sum(dim=-1)  # [B, M]
    valid_rows = (row_sums_xy > tiny) & mask_a
    row_entropy_mean = (row_entropy * valid_rows.to(dtype)).sum(dim=-1) / (valid_rows.to(dtype).sum(dim=-1).clamp_min(1.0))  # [B]

    # Column-wise normalized entropy
    q_col = P_xy / col_sums_xy.unsqueeze(-2)  # [B, M, N]
    col_entropy = -(q_col.clamp_min(tiny) * q_col.clamp_min(tiny).log()).sum(dim=-2)  # [B, N]
    valid_cols = (col_sums_xy > tiny) & mask_b
    col_entropy_mean = (col_entropy * valid_cols.to(dtype)).sum(dim=-1) / (valid_cols.to(dtype).sum(dim=-1).clamp_min(1.0))  # [B]

    # ====================================
    # Monotonicity corr & mapping smoothness
    # ====================================
    if M > 1:
        i_norm = torch.arange(M, device=device, dtype=dtype) / (M - 1)
    else:
        i_norm = torch.zeros(M, device=device, dtype=dtype)
    if N > 1:
        j_norm = torch.arange(N, device=device, dtype=dtype) / (N - 1)
    else:
        j_norm = torch.zeros(N, device=device, dtype=dtype)

    mu = (P_xy * j_norm.view(1, 1, N)).sum(dim=-1) / row_sums_xy  # [B, M]
    x = i_norm.view(1, M).expand(B, M)  # [B, M]
    w = (row_sums_xy - tiny) * mask_a.to(dtype)  # [B, M]
    wsum = w.sum(dim=-1).clamp_min(tiny)  # [B]

    x_mean = (w * x).sum(dim=-1) / wsum
    y_mean = (w * mu).sum(dim=-1) / wsum
    x_center = x - x_mean.unsqueeze(-1)
    y_center = mu - y_mean.unsqueeze(-1)

    cov = (w * x_center * y_center).sum(dim=-1) / wsum
    varx = (w * x_center.pow(2)).sum(dim=-1) / wsum
    vary = (w * y_center.pow(2)).sum(dim=-1) / wsum
    monotonicity_corr = cov / (torch.sqrt(varx * vary) + tiny)  # [B]

    valid_pairs = (row_sums_xy[..., :-1] > tiny) & (row_sums_xy[..., 1:] > tiny) & mask_a[..., :-1] & mask_a[..., 1:]
    mu_diff = (mu[..., 1:] - mu[..., :-1]).abs()
    mapping_tv = (mu_diff * valid_pairs.to(dtype)).sum(dim=-1) / (valid_pairs.to(dtype).sum(dim=-1).clamp_min(1.0))  # [B]

    # ======================
    # Diagonal band measures
    # ======================
    band = (i_norm.view(M, 1) - j_norm.view(1, N)).abs() <= float(delta)  # [M, N]
    band = band.to(P_xy.dtype).view(1, M, N)
    diag_band_mass = (P_xy * band).sum(dim=(-2, -1))  # [B]
    diag_band_mass_ratio = diag_band_mass / (mass_total + tiny)

    # ============================================
    # UOT objective J = <P,C> + eps*Σ p(log p -1)
    #                 + λ1 * KL(r||a) + λ2 * KL(c||b)
    # with KL(x||y) = Σ x(log(x/y)) - x + y   (masked)
    # ============================================
    def _kl_div(x, y, mask):
        """
        x,y: [B, K] nonnegative; mask: [B, K] bool.
        Returns: [B]
        """
        x_ = (x * mask.to(x.dtype)).clamp_min(0.0)
        y_ = (y * mask.to(y.dtype)).clamp_min(tiny)
        term = x_ * (torch.log(x_.clamp_min(tiny)) - torch.log(y_)) - x_ + y_
        return term.sum(dim=-1)

    def _uot_objective(P, C, a_row, a_col, mask_row, mask_col):
        # masks
        row_mask = mask_row
        col_mask = mask_col
        joint_mask = row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)
        Pm = torch.where(joint_mask, P, 0.0)
        Cm = torch.where(joint_mask, C, 1e6)

        # cost term
        cost_term = (Pm * Cm).sum(dim=(-2, -1))

        # entropic term: eps * Σ p (log p - 1)
        ent_term = reg * (Pm.clamp_min(tiny) * (Pm.clamp_min(tiny).log() - 1.0)).sum(dim=(-2, -1))

        # marginals
        r = Pm.sum(dim=-1)  # [B, M]
        c = Pm.sum(dim=-2)  # [B, N]

        # KL terms (masked)
        kl_r = _kl_div(r, torch.where(row_mask, a_row, 0.0), row_mask)
        kl_c = _kl_div(c, torch.where(col_mask, a_col, 0.0), col_mask)

        return cost_term + ent_term + lambda1 * kl_r + lambda2 * kl_c

    # XY objective
    uot_objective_xy = _uot_objective(P=transport_plan, C=cost_matrix, a_row=a, a_col=b, mask_row=mask_a, mask_col=mask_b)  # [B]

    # XX / YY objectives (if provided); if None, return NaN
    if (plan_xx is not None) and (cost_xx is not None):
        # reuse a, mask_a for both row/col in XX
        uot_objective_xx = _uot_objective(P=plan_xx, C=cost_xx, a_row=a, a_col=a, mask_row=mask_a, mask_col=mask_a)
    else:
        uot_objective_xx = torch.full_like(uot_objective_xy, float("nan"))

    if (plan_yy is not None) and (cost_yy is not None):
        # reuse b, mask_b for both row/col in YY
        uot_objective_yy = _uot_objective(P=plan_yy, C=cost_yy, a_row=b, a_col=b, mask_row=mask_b, mask_col=mask_b)
    else:
        uot_objective_yy = torch.full_like(uot_objective_xy, float("nan"))

    # Sinkhorn divergence: J_xy - 0.5(J_xx + J_yy)
    sinkhorn_divergence = uot_objective_xy - 0.5 * (uot_objective_xx + uot_objective_yy)

    return {
        # Metrics
        "transport_cost": transport_cost,  # <P_xy, C_xy>
        "mean_cosine": mean_cosine,  # sum P_xy*(1 - C_xy) / sum P_xy
        "mass_total": mass_total,  # sum P_xy
        "coverage_ratio": coverage_ratio,  # sum P_xy / min(sum a, sum b)
        "monotonicity_corr": monotonicity_corr,  # corr(i_norm, mu_i) weighted by row mass
        "mapping_tv": mapping_tv,  # mean |mu_{i+1}-mu_i|
        "row_entropy_mean": row_entropy_mean,  # avg row entropy over non-empty rows
        "col_entropy_mean": col_entropy_mean,  # avg col entropy over non-empty cols
        "diag_band_mass": diag_band_mass,  # mass inside diagonal band
        "diag_band_mass_ratio": diag_band_mass_ratio,  # band_mass / total_mass
        # UOT objectives / Sinkhorn divergence
        "uot_objective_xy": uot_objective_xy,  # J(X,Y)
        "uot_objective_xx": uot_objective_xx,  # J(X,X) or NaN
        "uot_objective_yy": uot_objective_yy,  # J(Y,Y) or NaN
        "sinkhorn_divergence": sinkhorn_divergence,  # J_xy - 1/2(J_xx + J_yy)
    }
