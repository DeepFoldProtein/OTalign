from dataclasses import dataclass
from typing import Callable, Optional, cast

import numpy as np
from numba import njit
from scipy.optimize import brentq, minimize_scalar

from otalign.align.uot_alignment import _dp_core_numba


def uot_alignment_path(S: np.ndarray, goD: np.ndarray, geD: np.ndarray, goI: np.ndarray, geI: np.ndarray):
    ge_q = np.concatenate(([0.0], geD))
    go_q = np.concatenate(([0.0], goD))
    ge_t = np.concatenate(([0.0], geI))
    go_t = np.concatenate(([0.0], goI))

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

    return path, {"best_score": best_score}


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

    # Allocate forward
    F_M = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    F_D = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    F_I = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)

    # ---- Forward init ----

    # global: classic boundaries
    F_M[0, 0] = 0.0
    # first column
    for i in range(1, n + 1):
        F_D[i, 0] = 0.0  # F_D[i - 1, 0] + geD[i - 1]
    # first row
    for j in range(1, m + 1):
        F_I[0, j] = 0.0  # F_I[0, j - 1] + geI[j - 1]

    # ---- Forward DP ----
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # M(i,j)
            t_prev = lse3(F_M[i - 1, j - 1], F_D[i - 1, j - 1], F_I[i - 1, j - 1])
            val_M = E_M[i - 1, j - 1] + t_prev
            F_M[i, j] = val_M

            # D(i,j) : from M open+ext at i, or D extend at i
            d_open = F_M[i - 1, j] + goD[i - 1] + geD[i - 1]
            d_extend = F_D[i - 1, j] + geD[i - 1]
            val_D = lse2(d_open, d_extend)
            F_D[i, j] = val_D

            # I(i,j) : from M open+ext at j, or I extend at j
            i_open = F_M[i, j - 1] + goI[j - 1] + geI[j - 1]
            i_extend = F_I[i, j - 1] + geI[j - 1]
            val_I = lse2(i_open, i_extend)
            F_I[i, j] = val_I

    # ---- logZ ----

    # global: only (n,m)
    logZ = lse3(F_M[n, m], F_D[n, m], F_I[n, m])

    # Allocate backward
    B_M = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    B_D = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)
    B_I = np.full((n + 1, m + 1), -np.inf, dtype=np.float64)

    # ---- Backward init (same as before) ----
    B_M[n, m] = 0.0
    B_D[n, m] = 0.0
    B_I[n, m] = 0.0

    # ---- Backward DP (from bottom-right to top-left) ----
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            # *** 핵심 가드: 글로벌에서는 (n,m)을 덮어쓰지 않음 ***
            if (i == n) and (j == m):
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

            B_M[i, j] = lse3(t1, t2, t3)

            # D(i,j) -> { D(i+1,j), M(i+1,j+1) }
            u1 = -np.inf
            u2 = -np.inf
            if (i + 1) <= n:
                u1 = geD[i] + B_D[i + 1, j]
            if (i + 1) <= n and (j + 1) <= m:
                u2 = E_M[i, j] + B_M[i + 1, j + 1]

            B_D[i, j] = lse2(u1, u2)

            # I(i,j) -> { I(i,j+1), M(i+1,j+1) }
            v1 = -np.inf
            v2 = -np.inf
            if (j + 1) <= m:
                v1 = geI[j] + B_I[i, j + 1]
            if (i + 1) <= n and (j + 1) <= m:
                v2 = E_M[i, j] + B_M[i + 1, j + 1]

            B_I[i, j] = lse2(v1, v2)

    return F_M, F_D, F_I, B_M, B_D, B_I, logZ


@njit
def get_antidiagonal_indices(k, m, n):
    """Helper to get anti-diagonal indices as a NumPy array."""
    start_i = max(0, k - (n - 1))
    end_i = min(m - 1, k)
    size = end_i - start_i + 1
    if size <= 0:
        return np.empty((0, 2), dtype=np.int64)
    arr = np.empty((size, 2), dtype=np.int64)
    for i_idx, i in enumerate(range(start_i, end_i + 1)):
        arr[i_idx, 0] = i
        arr[i_idx, 1] = k - i
    return arr


@njit
def _diag_avg(values, indices, use_i_axis):
    """Numba-friendly version of the diagonal averaging."""
    # `indices` is now a NumPy array.
    total = 0.0
    count = 0
    if use_i_axis:
        for i in range(indices.shape[0]):
            idx_val = indices[i, 0]
            if 0 <= idx_val < len(values):
                total += values[idx_val]
                count += 1
    else:
        for i in range(indices.shape[0]):
            idx_val = indices[i, 1]
            if 0 <= idx_val < len(values):
                total += values[idx_val]
                count += 1

    if count == 0:
        # This will be caught by Numba and raised as a Python exception.
        raise ValueError("No valid indices for position-dependent parameter on this diagonal.")
    return total / count


@njit
def calculate_lyapunov_pressure(score_matrix, g_od, g_ed, g_oi, g_ei, mu, gamma):
    """
    Top Lyapunov exponent (finite-length pressure) for position-dependent model.
    Numba-optimized version.

    Args:
        score_matrix: m x n ARRAY of *scores* s_ij (positive good; can be real-valued).
        g_od, g_ed, g_oi, g_ei: 1D arrays for gap penalties (position-dependent).
        mu:    match fugacity (adds to match log-score).
        gamma: tilt parameter for large deviations (gamma=1 is physical).
    Returns:
        float pressure P_K(mu, gamma) = (1/K) * log || prod_k T_k(mu, gamma) ||_1
    """
    s = score_matrix
    m, n = s.shape
    K = m + n - 1

    v = np.ones(3, dtype=np.float64)
    sum_log_c = 0.0
    tiny = 1e-300

    for k in range(K):
        idx = get_antidiagonal_indices(k, m, n)
        if idx.shape[0] == 0:
            continue
        II, JJ = idx.T

        # --- Match moment on this diagonal (stable log-mean-exp) ---
        # Numba doesn't support advanced indexing with two arrays (s[II, JJ]).
        # We must loop explicitly.
        x = np.empty(II.shape[0], dtype=np.float64)
        for i in range(II.shape[0]):
            x[i] = gamma * s[II[i], JJ[i]]

        xmax = np.max(x)
        lme = xmax + np.log(np.mean(np.exp(x - xmax)))
        Ea_k = np.exp(lme + mu)

        # --- Gap entries (diagonal means) ---
        alpha_D_k = _diag_avg(g_od, idx, True)
        beta_D_k = _diag_avg(g_ed, idx, True)
        alpha_I_k = _diag_avg(g_oi, idx, False)
        beta_I_k = _diag_avg(g_ei, idx, False)

        T_MD_k = np.exp(gamma * alpha_D_k)
        T_DD_k = np.exp(gamma * beta_D_k)
        T_MI_k = np.exp(gamma * alpha_I_k)
        T_II_k = np.exp(gamma * beta_I_k)

        # 3x3 slice
        T_k = np.array([[Ea_k, T_MD_k, T_MI_k], [Ea_k, T_DD_k, 0.0], [Ea_k, 0.0, T_II_k]], dtype=np.float64)

        # Power step with L1 renormalization (positive vectors -> stable)
        v = T_k @ v
        c = np.sum(np.abs(v))
        if not np.isfinite(c) or c < tiny:
            v[:] = 1.0 / 3.0
            continue
        v /= c
        sum_log_c += np.log(c)

    return (sum_log_c / K) if K > 0 else 0.0


def find_critical_fugacity_lyapunov(score_matrix, gaps, mu_bracket, gamma=1.0, tol=1e-6) -> float:
    """
    Find mu_c by solving F(mu) = P(mu, gamma=1) = 0.
    """
    mu_L, mu_U = map(float, mu_bracket)

    # Ensure inputs are numpy arrays for Numba function
    s_mat = np.asarray(score_matrix, dtype=np.float64)
    g_od = np.asarray(gaps["gap_open_del"], dtype=np.float64)
    g_ed = np.asarray(gaps["gap_ext_del"], dtype=np.float64)
    g_oi = np.asarray(gaps["gap_open_ins"], dtype=np.float64)
    g_ei = np.asarray(gaps["gap_ext_ins"], dtype=np.float64)

    # Numba requires 1D arrays, but scalars become 0D. Flatten to 1D.
    if g_od.ndim == 0:
        g_od = g_od.flatten()
    if g_ed.ndim == 0:
        g_ed = g_ed.flatten()
    if g_oi.ndim == 0:
        g_oi = g_oi.flatten()
    if g_ei.ndim == 0:
        g_ei = g_ei.flatten()

    def F(mu):
        return calculate_lyapunov_pressure(s_mat, g_od, g_ed, g_oi, g_ei, mu, gamma)

    p_L, p_U = F(mu_L), F(mu_U)
    if not np.isfinite(p_L) or not np.isfinite(p_U) or np.sign(p_L) == np.sign(p_U):
        raise ValueError(f"Invalid mu bracket [{mu_L}, {mu_U}] for mu_c: F(L)={p_L:.6g}, F(U)={p_U:.6g} (need F(L)<0<F(U)).")

    # Use Brent's method for faster and more robust root finding.
    return cast(float, brentq(F, mu_L, mu_U, xtol=tol))


def find_lambda_lyapunov(mu, score_matrix, gaps, tol=1e-6) -> float:
    """
    Find lambda(mu) by solving P(mu, gamma) = 0 for mu <= mu_c.
    Bracket: gamma in [0, 1], since P(mu,0) > 0 and P(mu,1) <= 0 in subcritical regime.
    """

    g_od = np.asarray(gaps["gap_open_del"], dtype=np.float64)
    g_ed = np.asarray(gaps["gap_ext_del"], dtype=np.float64)
    g_oi = np.asarray(gaps["gap_open_ins"], dtype=np.float64)
    g_ei = np.asarray(gaps["gap_ext_ins"], dtype=np.float64)

    def P_gamma(gamma):
        return calculate_lyapunov_pressure(score_matrix, g_od, g_ed, g_oi, g_ei, mu, gamma)

    gamma_L, gamma_U = 0.0, 1.0

    # Evaluate ends once to check validity of bracket
    p0 = P_gamma(gamma_L)
    p1 = P_gamma(gamma_U)

    if not np.isfinite(p0) or not np.isfinite(p1):
        raise ValueError(f"Non-finite pressure at bracket ends: P(mu,0)={p0}, P(mu,1)={p1}")

    if p1 > 0.0:
        # Supercritical case (mu > mu_c), lambda is not well-defined as a root in [0,1].
        raise ValueError(f"Cannot find lambda for supercritical mu. F(mu)=P(mu,1)={p1:.6g} > 0 at mu={mu:.6g}.")

    # Brent's method requires endpoints to have different signs.
    # The check for p1 > 0.0 handles the supercritical case.
    # This handles the unusual case where P(mu,0) < 0.
    if np.sign(p0) == np.sign(p1):
        raise ValueError(f"Root finding for lambda failed. Bracket values P(mu,0)={p0:.6g} and P(mu,1)={p1:.6g} do not straddle zero.")

    # Use Brent's method for faster and more robust root finding.
    return cast(float, brentq(P_gamma, gamma_L, gamma_U, xtol=tol))


def fd_second_5pt(f: Callable[[float], float], x: float, h: float) -> float:
    """5-point central stencil for the 2nd derivative.
    f''(x) ≈ [-f(x+2h) + 16 f(x+h) - 30 f(x) + 16 f(x-h) - f(x-2h)] / (12 h^2)
    """
    f_p2 = f(x + 2 * h)
    f_p1 = f(x + h)
    f_0 = f(x)
    f_m1 = f(x - h)
    f_m2 = f(x - 2 * h)
    return (-f_p2 + 16.0 * f_p1 - 30.0 * f_0 + 16.0 * f_m1 - f_m2) / (12.0 * h * h)


def fd_third_5pt(f: Callable[[float], float], x: float, h: float) -> float:
    """5-point central stencil for the 3rd derivative.
    f'''(x) ≈ [f(x-2h) - 2 f(x-h) + 2 f(x+h) - f(x+2h)] / (2 h^3)
    """
    f_p2 = f(x + 2 * h)
    f_p1 = f(x + h)
    f_m1 = f(x - h)
    f_m2 = f(x - 2 * h)
    return (f_m2 - 2.0 * f_m1 + 2.0 * f_p1 - f_p2) / (2.0 * h**3)


def fd_fourth_5pt(f: Callable[[float], float], x: float, h: float) -> float:
    """5-point central stencil for the 4th derivative.
    f''''(x) ≈ [f(x-2h) - 4 f(x-h) + 6 f(x) - 4 f(x+h) + f(x+2h)] / (h^4)
    """
    f_p2 = f(x + 2 * h)
    f_p1 = f(x + h)
    f_0 = f(x)
    f_m1 = f(x - h)
    f_m2 = f(x - 2 * h)
    return (f_m2 - 4.0 * f_m1 + 6.0 * f_0 - 4.0 * f_p1 + f_p2) / (h**4)


def golden_maximize(g: Callable[[float], float], a: float, b: float, tol: float = 1e-4, max_iter: int = 200) -> tuple[float, float]:
    """Maximize g on [a,b] using scipy's minimize_scalar. Returns (x*, g(x*)).
    Assumes g is unimodal on the interval."""
    # We use minimize_scalar to find the maximum by minimizing the negative of the function.
    res = minimize_scalar(lambda x: -g(x), bounds=(a, b), method="bounded", options={"xatol": tol, "maxiter": max_iter})

    if not res.success:
        # Depending on requirements, could warn or raise here.
        # For now, just return the found value.
        pass

    return res.x, -res.fun


@dataclass
class PeakSearchConfig:
    bracket_radius: float = 1.0  # search window around mu_c
    h: float = 0.05  # finite-difference step
    golden_tol: float = 1e-4  # tolerance for golden-section
    newton_tol: float = 1e-6  # stop when |P'''| < tol
    max_newton_iter: int = 12


def estimate_var_peak_by_pressure(
    pressure_func: Callable[[float], float],
    mu_c: float,
    cfg: PeakSearchConfig = PeakSearchConfig(),
) -> tuple[float, float]:
    """Method 1: Fast proxy peak via maximizing the 2nd derivative of P(mu).
    Returns (mu_hat, proxy_second_derivative_value)."""
    a = mu_c - cfg.bracket_radius
    b = mu_c + cfg.bracket_radius

    def second_deriv(mu: float) -> float:
        return fd_second_5pt(pressure_func, mu, cfg.h)

    mu_hat, sec_val = golden_maximize(second_deriv, a, b, tol=cfg.golden_tol)
    return mu_hat, sec_val


def newton_var_peak_by_pressure(
    pressure_func: Callable[[float], float],
    mu0: float,
    cfg: PeakSearchConfig = PeakSearchConfig(),
    hard_bracket: Optional[tuple[float, float]] = None,
) -> tuple[float, int]:
    """Method 2: Find root of P'''(mu)=0 using Brent's method.
    Returns (mu_star, iters=1 for compatibility). Optionally clamps to 'hard_bracket'."""

    def third_deriv(mu: float) -> float:
        return fd_third_5pt(pressure_func, mu, cfg.h)

    # Define search bracket for the root of P'''
    # If a hard_bracket is provided, use it. Otherwise, create one around mu0.
    if hard_bracket:
        a, b = hard_bracket
    else:
        # Heuristic bracket around the initial guess mu0
        a, b = mu0 - cfg.bracket_radius / 2, mu0 + cfg.bracket_radius / 2

    try:
        # Brent's method requires the function values at the endpoints to have opposite signs.
        fa, fb = third_deriv(a), third_deriv(b)
        if np.sign(fa) == np.sign(fb):
            # If not, we can try to expand the bracket or just fail.
            # For now, we'll raise an error as this indicates an issue with the bracket.
            raise ValueError(f"Root for P''' not bracketed in [{a}, {b}]. f(a)={fa}, f(b)={fb}")

        mu_star, r = brentq(third_deriv, a, b, xtol=cfg.newton_tol, full_output=True)
        return mu_star, r.iterations
    except (ValueError, RuntimeError):
        # Fallback or error handling if brentq fails
        # For now, just return the initial guess. A warning could be logged.
        # print(f"Warning: Newton-like peak finding failed: {e}. Returning mu0.")
        return mu0, 0
