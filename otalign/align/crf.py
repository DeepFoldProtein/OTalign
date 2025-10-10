import numpy as np

from otalign.align.uot_alignment import _dp_core_numba


def uot_alignment_path(C: np.ndarray, phi: np.ndarray, psi: np.ndarray, eps: float, tau: float, mu: float, rho_min: float | None = 1e-6):
    # Match score and gap hazards
    S = (phi[:, None] + psi[None, :] - C) / eps + mu
    rho_D = np.exp(-phi / tau)
    rho_I = np.exp(-psi / tau)

    # Practical "gotchas" and fixes: Keep hazards in (0,1)
    if rho_min:
        rho_D = np.clip(rho_D, rho_min, 1 - rho_min)
        rho_I = np.clip(rho_I, rho_min, 1 - rho_min)
    else:
        rho_D = rho_D / (1 + rho_D)
        rho_I = rho_I / (1 + rho_I)

    # Affine components
    g_ext_D = np.log(rho_D)
    g_open_D = np.log1p(-rho_D)
    g_ext_I = np.log(rho_I)
    g_open_I = np.log1p(-rho_I)

    ge_q = np.concatenate(([0.0], g_ext_D))
    go_q = np.concatenate(([0.0], g_open_D))
    ge_t = np.concatenate(([0.0], g_ext_I))
    go_t = np.concatenate(([0.0], g_open_I))

    _mode_flag = 3
    _band = 0

    M, X, Y, ptrM, ptrX, ptrY, i, j, state, best_score = _dp_core_numba(S, go_q, ge_q, go_t, ge_t, _mode_flag, _band)

    # Traceback
    path = []
    while i > 0 or j > 0:
        if state == 0:  # M
            if i == 0 or j == 0:
                break
            prev_state = int(ptrM[i, j])
            path.append((i - 1, j - 1, "M"))
            i -= 1
            j -= 1
            state = prev_state
        elif state == 1:  # X => insertion (gap in target)
            if i == 0:
                break
            prev = int(ptrX[i, j])
            path.append((i - 1, j - 1, "I"))
            i -= 1
            state = 0 if prev == 0 else 1
        else:  # Y => deletion (gap in query)
            if j == 0:
                break
            prev = int(ptrY[i, j])
            path.append((i - 1, j - 1, "D"))
            j -= 1
            state = 0 if prev == 0 else 2

        # glocal early stop (q2t): stop at edges on open side
        if _mode_flag == 1 and (i == 0 or j == 0):
            break

    path.reverse()

    return path, {"best_score": best_score, "score": S, "rho_D": rho_D, "rho_I": rho_I, "goD": g_open_D, "geD": g_ext_D, "goI": g_open_I, "geI": g_ext_I}


import numpy as np
from numba import njit


# ---------- numerics ----------
@njit(inline="always")
def lse2(x, y):
    m = x if x > y else y
    if np.isneginf(m):
        return m
    return m + np.log1p(np.exp((x - m) + (y - m)))


@njit(inline="always")
def lse3(x, y, z):
    # stable logsumexp of 3 terms
    m = x
    if y > m:
        m = y
    if z > m:
        m = z
    if np.isneginf(m):
        return m
    return m + np.log(np.exp(x - m) + np.exp(y - m) + np.exp(z - m))


@njit(inline="always")
def lse4(a, b, c, d):
    # for local logZ accumulation convenience
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    if d > m:
        m = d
    if np.isneginf(m):
        return m
    return m + np.log(np.exp(a - m) + np.exp(b - m) + np.exp(c - m) + np.exp(d - m))


# ---------- core forward/backward ----------
@njit
def crf_forward_backward(E_M, goD, geD, goI, geI):
    """
    Compute Forward(F), Backward(B) and logZ for 3-state alignment CRF.

    Inputs
    ------
    E_M : (n, m) float64    # match emission: S + mu (already offset-added)
    goD : (n,) float64      # deletion open penalty at i (log(1-rho_D(i)))
    geD : (n,) float64      # deletion extend penalty at i (log rho_D(i))
    goI : (m,) float64      # insertion open penalty at j (log(1-rho_I(j)))
    geI : (m,) float64      # insertion extend penalty at j (log rho_I(j))

    Returns
    -------
    F_M, F_D, F_I : (n+1, m+1) float64  # forward log-weights ending in M/D/I at (i,j)
    B_M, B_D, B_I : (n+1, m+1) float64  # backward log-weights from (i,j)
    logZ          : float64
    """
    n, m = E_M.shape
    local = False

    # Allocate forward
    F_M = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    F_D = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    F_I = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)

    # ---- Forward init ----
    if local:
        # local: anywhere can start (0-clamp will also apply inside recurrences)
        F_M[0, 0] = 0.0
        F_D[0, 0] = 0.0
        F_I[0, 0] = 0.0
        for j in range(1, m + 1):
            F_I[0, j] = 0.0
        for i in range(1, n + 1):
            F_D[i, 0] = 0.0
    else:
        # global: classic boundaries
        F_M[0, 0] = 0.0
        # first column: deletions extend only
        for i in range(1, n + 1):
            F_D[i, 0] = 0.0  # F_D[i - 1, 0] + geD[i - 1]
        # first row: insertions extend only
        for j in range(1, m + 1):
            F_I[0, j] = 0.0  # F_I[0, j - 1] + geI[j - 1]

    # ---- Forward DP ----
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # M(i,j)
            t_prev = lse3(F_M[i - 1, j - 1], F_D[i - 1, j - 1], F_I[i - 1, j - 1])
            val_M = E_M[i - 1, j - 1] + t_prev
            if local:
                val_M = lse2(val_M, 0.0)
            F_M[i, j] = val_M

            # D(i,j) : from M open+ext at i, or D extend at i
            d_open = F_M[i - 1, j] + goD[i - 1] + geD[i - 1]
            d_extend = F_D[i - 1, j] + geD[i - 1]
            val_D = lse2(d_open, d_extend)
            if local:
                val_D = lse2(val_D, 0.0)
            F_D[i, j] = val_D

            # I(i,j) : from M open+ext at j, or I extend at j
            i_open = F_M[i, j - 1] + goI[j - 1] + geI[j - 1]
            i_extend = F_I[i, j - 1] + geI[j - 1]
            val_I = lse2(i_open, i_extend)
            if local:
                val_I = lse2(val_I, 0.0)
            F_I[i, j] = val_I

    # ---- logZ ----
    if local:
        # sum over all endpoints (add a virtual sink with 0-cost)
        logZ = -np.inf
        for i in range(n + 1):
            for j in range(m + 1):
                logZ = lse4(logZ, F_M[i, j], F_D[i, j], F_I[i, j])
    else:
        # global: only (n,m)
        logZ = lse3(F_M[n, m], F_D[n, m], F_I[n, m])

    # Allocate backward
    B_M = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    B_D = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    B_I = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)

    # ---- Backward init (same as before) ----
    if local:
        for i in range(n + 1):
            for j in range(m + 1):
                B_M[i, j] = 0.0
                B_D[i, j] = 0.0
                B_I[i, j] = 0.0
    else:
        B_M[n, m] = 0.0
        B_D[n, m] = 0.0
        B_I[n, m] = 0.0

    # ---- Backward DP (from bottom-right to top-left) ----
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            # *** 핵심 가드: 글로벌에서는 (n,m)을 덮어쓰지 않음 ***
            if (not local) and (i == n) and (j == m):
                continue  # keep B_*[n,m] == 0.0

            # M(i,j) -> { M(i+1,j+1), D(i+1,j), I(i,j+1) }
            t1 = -np.inf
            t2 = -np.inf
            t3 = -np.inf
            if (i + 1) <= n and (j + 1) <= m:
                t1 = E_M[i, j] + B_M[i + 1, j + 1]
            if (i + 1) <= n:
                t2 = goD[i] + geD[i] + B_D[i + 1, j]
            if (j + 1) <= m:
                t3 = goI[j] + geI[j] + B_I[i, j + 1]

            if local:
                # 로컬: 종료 가능하므로 후속이 없으면 0 유지
                if np.isneginf(t1) and np.isneginf(t2) and np.isneginf(t3):
                    # keep B_M[i,j] as is (already 0.0)
                    pass
                else:
                    B_M[i, j] = lse3(t1, t2, t3)
            else:
                B_M[i, j] = lse3(t1, t2, t3)

            # D(i,j) -> { D(i+1,j), M(i+1,j+1) }
            u1 = -np.inf
            u2 = -np.inf
            if (i + 1) <= n:
                u1 = geD[i] + B_D[i + 1, j]
            if (i + 1) <= n and (j + 1) <= m:
                u2 = E_M[i, j] + B_M[i + 1, j + 1]
            if local:
                if np.isneginf(u1) and np.isneginf(u2):
                    pass
                else:
                    B_D[i, j] = lse2(u1, u2)
            else:
                B_D[i, j] = lse2(u1, u2)

            # I(i,j) -> { I(i,j+1), M(i+1,j+1) }
            v1 = -np.inf
            v2 = -np.inf
            if (j + 1) <= m:
                v1 = geI[j] + B_I[i, j + 1]
            if (i + 1) <= n and (j + 1) <= m:
                v2 = E_M[i, j] + B_M[i + 1, j + 1]
            if local:
                if np.isneginf(v1) and np.isneginf(v2):
                    pass
                else:
                    B_I[i, j] = lse2(v1, v2)
            else:
                B_I[i, j] = lse2(v1, v2)

    return F_M, F_D, F_I, B_M, B_D, B_I, logZ


def get_antidiagonal_indices(k, m, n):
    start_i = max(0, k - (n - 1))
    end_i = min(m - 1, k)
    return [(i, k - i) for i in range(start_i, end_i + 1)]


def _diag_avg(values, indices, use_i_axis):
    vals = []
    if use_i_axis:
        for i, j in indices:
            if 0 <= i < len(values):
                vals.append(values[i])
    else:
        for i, j in indices:
            if 0 <= j < len(values):
                vals.append(values[j])
    if not vals:
        raise ValueError("No valid indices for position-dependent parameter on this diagonal.")
    return float(np.mean(vals, dtype=np.float64))


def calculate_lyapunov_pressure(score_matrix, gaps, mu, gamma):
    """
    Top Lyapunov exponent (finite-length pressure) for position-dependent model.

    Args:
        score_matrix: m x n ARRAY of *scores* s_ij (positive good; can be real-valued).
        gaps: dict with keys:
              'gap_open_del', 'gap_ext_del', 'gap_open_ins', 'gap_ext_ins'
              Each can be scalar or 1D array (position-dependent).
        mu:    match fugacity (adds to match log-score).
        gamma: tilt parameter for large deviations (gamma=1 is physical).
    Returns:
        float pressure P_K(mu, gamma) = (1/K) * log || prod_k T_k(mu, gamma) ||_1
    """
    s = np.asarray(score_matrix, dtype=np.float64)
    m, n = s.shape
    K = m + n - 1

    # Fetch (possibly position-dependent) penalties
    g_od = gaps["gap_open_del"]
    g_ed = gaps["gap_ext_del"]
    g_oi = gaps["gap_open_ins"]
    g_ei = gaps["gap_ext_ins"]

    v = np.ones(3, dtype=np.float64)
    sum_log_c = 0.0
    tiny = 1e-300

    for k in range(K):
        idx = get_antidiagonal_indices(k, m, n)
        if not idx:
            continue
        II, JJ = np.array(idx, dtype=int).T  # shape (len_diag,)

        # --- Match moment on this diagonal (stable log-mean-exp) ---
        # M_k = mean_{(i,j) in diag k} exp(gamma * s_ij) * exp(mu)
        x = gamma * s[II, JJ]
        # log-mean-exp = log( mean(exp(x)) ) = logsumexp(x) - log(len_diag)
        xmax = np.max(x)
        lme = xmax + np.log(np.mean(np.exp(x - xmax)))
        Ea_k = np.exp(lme + mu)  # strictly > 0

        # --- Gap entries (diagonal means) ---
        if np.isscalar(g_od):
            alpha_D_k = float(g_od)
        else:
            alpha_D_k = _diag_avg(np.asarray(g_od, dtype=np.float64), idx, use_i_axis=True)

        if np.isscalar(g_ed):
            beta_D_k = float(g_ed)
        else:
            beta_D_k = _diag_avg(np.asarray(g_ed, dtype=np.float64), idx, use_i_axis=True)

        if np.isscalar(g_oi):
            alpha_I_k = float(g_oi)
        else:
            alpha_I_k = _diag_avg(np.asarray(g_oi, dtype=np.float64), idx, use_i_axis=False)

        if np.isscalar(g_ei):
            beta_I_k = float(g_ei)
        else:
            beta_I_k = _diag_avg(np.asarray(g_ei, dtype=np.float64), idx, use_i_axis=False)

        T_MD_k = np.exp(gamma * alpha_D_k)
        T_DD_k = np.exp(gamma * beta_D_k)
        T_MI_k = np.exp(gamma * alpha_I_k)
        T_II_k = np.exp(gamma * beta_I_k)

        # 3x3 slice
        T_k = np.array([[Ea_k, T_MD_k, T_MI_k], [Ea_k, T_DD_k, 0.0], [Ea_k, 0.0, T_II_k]], dtype=np.float64)

        # Power step with L1 renormalization (positive vectors -> stable)
        v = T_k @ v
        c = float(np.sum(np.abs(v)))
        if not np.isfinite(c) or c < tiny:
            # stabilize (shouldn't happen with positive entries)
            v[:] = 1.0 / 3.0
            continue
        v /= c
        sum_log_c += np.log(c)

    return (sum_log_c / K) if K > 0 else 0.0


def find_critical_fugacity_lyapunov(score_matrix, gaps, mu_bracket, tol=1e-6):
    """
    Find mu_c by solving F(mu) = P(mu, gamma=1) = 0.
    """
    mu_L, mu_U = map(float, mu_bracket)

    def F(mu):
        return calculate_lyapunov_pressure(score_matrix, gaps, mu, 1.0)

    p_L, p_U = F(mu_L), F(mu_U)
    if not np.isfinite(p_L) or not np.isfinite(p_U) or np.sign(p_L) == np.sign(p_U):
        raise ValueError(f"Invalid mu bracket [{mu_L}, {mu_U}] for mu_c: F(L)={p_L:.6g}, F(U)={p_U:.6g} (need F(L)<0<F(U)).")

    while (mu_U - mu_L) > tol:
        mu_M = 0.5 * (mu_L + mu_U)
        p_M = F(mu_M)
        if p_M < 0.0:
            mu_L = mu_M
        else:
            mu_U = mu_M
    return 0.5 * (mu_L + mu_U)
