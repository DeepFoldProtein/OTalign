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

    return path, {"best_score": best_score, "score": S}
