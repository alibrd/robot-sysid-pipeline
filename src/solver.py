"""Parameter identification solvers: OLS, WLS, bounded least-squares,
and physically-constrained least-squares.

The constrained path ("lmi") enforces pseudo-inertia PSD per link, which
is the correct necessary-and-sufficient condition for physical consistency
(see Wensing, Kim & Slotine 2018).  The solver wraps the LS problem inside
an SLSQP optimisation with per-link eigenvalue constraints on the 4×4
pseudo-inertia matrix J.

The Cholesky path ("cholesky") reparameterises each link's pseudo-inertia
as J = L Lᵀ where L is lower-triangular, guaranteeing J ≽ 0 by
construction.  Optimisation is unconstrained in L-space (L-BFGS-B).

Constrained methods require a full 10-parameter block per moving link.  The
pipeline regressor model exposes that public contract for every backend.
"""
import logging
import numpy as np
from scipy.optimize import lsq_linear, minimize

from .feasibility import pseudo_inertia_matrix, pi_from_pseudo_inertia_matrix

logger = logging.getLogger("sysid_pipeline")


def solve_identification(W: np.ndarray,
                         tau_vec: np.ndarray,
                         solver: str = "ols",
                         bounds: tuple = None,
                         weights: np.ndarray = None,
                         nDoF: int = 0,
                         feasibility_method: str = "none",
                         P_mat: np.ndarray = None):
    """Solve the linear identification problem  tau = W pi.

    Parameters
    ----------
    W : (m, p) observation matrix (may be base-parameter-reduced)
    tau_vec : (m,) stacked torque measurements
    solver : "ols" | "wls" | "bounded_ls"
    bounds : (lb, ub) each (p,) for bounded_ls
    weights : (m,) for weighted least-squares (wls)
    nDoF : number of joints (needed for constrained identification)
    feasibility_method : "none" | "lmi" | "cholesky"
        When "lmi", the solver wraps the LS problem inside an SLSQP
        optimisation with physical feasibility constraints on the FULL
        parameter vector (10 per link).  When "cholesky", the solver
        reparameterises each link's pseudo-inertia as J = L Lᵀ,
        guaranteeing PSD by construction (unconstrained L-BFGS-B).
        Requires a full 10-parameter block per moving link.
    P_mat : (r, p_full) regrouping matrix, required when feasibility_method
        is not "none". Relates W_base @ pi_base = W_full @ pi_full via
        pi_base = P @ pi_full.  The optimisation is then over pi_full.

    Returns
    -------
    pi_hat : (p,) estimated parameters (base params when unconstrained,
             full params when constrained)
    residual : scalar residual norm
    info : dict with solver-specific information
    """
    m, p = W.shape
    logger.info("Solver: %s, matrix %dx%d, overdetermination %.1f:1",
                solver, m, p, m / max(p, 1))

    # --- Constrained identification (LMI) ----------------------------------
    if feasibility_method == "lmi" and nDoF > 0:
        if P_mat is None:
            logger.warning("Constrained identification requested but no regrouping "
                           "matrix P provided. Falling back to unconstrained solver.")
        else:
            return _solve_constrained(W, tau_vec, nDoF, feasibility_method,
                                      solver, weights, P_mat)

    # --- Cholesky reparameterisation ----------------------------------------
    if feasibility_method == "cholesky" and nDoF > 0:
        if P_mat is None:
            logger.warning("Cholesky identification requested but no regrouping "
                           "matrix P provided. Falling back to unconstrained solver.")
        else:
            return _solve_cholesky(W, tau_vec, nDoF, solver, weights, P_mat)

    # --- Standard solvers --------------------------------------------------
    if solver == "ols":
        pi_hat, residuals, rank, sv = np.linalg.lstsq(W, tau_vec, rcond=None)
        res_norm = np.linalg.norm(W @ pi_hat - tau_vec)
        logger.info("OLS: rank=%d, residual=%.6e", rank, res_norm)
        return pi_hat, res_norm, {"rank": rank, "singular_values": sv}

    elif solver == "wls":
        if weights is None:
            # Estimate weights from residual of OLS (iteratively-reweighted LS)
            pi_ols, _, _, _ = np.linalg.lstsq(W, tau_vec, rcond=None)
            resid = tau_vec - W @ pi_ols
            sigma2 = np.maximum(resid**2, 1e-16)
            weights = 1.0 / sigma2
            logger.info("WLS: weights estimated from OLS residuals (IRLS step).")
        sqrt_w = np.sqrt(weights)
        W_w = W * sqrt_w[:, None]
        tau_w = tau_vec * sqrt_w
        pi_hat, residuals, rank, sv = np.linalg.lstsq(W_w, tau_w, rcond=None)
        res_norm = np.linalg.norm(W @ pi_hat - tau_vec)
        logger.info("WLS: rank=%d, residual=%.6e", rank, res_norm)
        return pi_hat, res_norm, {"rank": rank, "singular_values": sv}

    elif solver == "bounded_ls":
        if bounds is None:
            lb = -np.inf * np.ones(p)
            ub = np.inf * np.ones(p)
        else:
            lb, ub = bounds
        result = lsq_linear(W, tau_vec, bounds=(lb, ub), method="bvls",
                             verbose=0)
        pi_hat = result.x
        res_norm = result.cost
        logger.info("Bounded LS: residual=%.6e, status=%d", res_norm, result.status)
        return pi_hat, res_norm, {"status": result.status, "message": result.message}

    else:
        raise ValueError(f"Unknown solver: {solver}")


def _solve_constrained(W_base, tau_vec, nDoF, feasibility_method, base_solver,
                       weights, P_mat):
    """Solve LS in the FULL parameter space with pseudo-inertia PSD constraints.

    The observation equation is  tau = W_base @ pi_base = W_base @ P @ pi_full.
    We optimise over pi_full (10*nDoF) with the constraint that each link's
    4×4 pseudo-inertia matrix J_i is positive semidefinite.
    """
    p_full = P_mat.shape[1]
    W_full = W_base @ P_mat  # (m, p_full)

    eps_eig = 1e-8

    # OLS on full-space as initial guess
    pi0, _, _, _ = np.linalg.lstsq(W_full, tau_vec, rcond=None)

    # Pre-compute for fast cost
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        W_w = W_full * sqrt_w[:, None]
        tau_w = tau_vec * sqrt_w
        WtW = W_w.T @ W_w
        Wt_tau = W_w.T @ tau_w
    else:
        WtW = W_full.T @ W_full
        Wt_tau = W_full.T @ tau_vec

    def cost(pi):
        r = W_full @ pi - tau_vec
        return 0.5 * np.dot(r, r)

    def grad(pi):
        return WtW @ pi - Wt_tau

    # Build pseudo-inertia PSD constraints per link
    constraints = []
    n_rigid = min(nDoF, p_full // 10)

    for i in range(n_rigid):
        base = 10 * i
        if base + 10 > p_full:
            break

        # Minimum eigenvalue of pseudo-inertia J >= eps_eig
        def _pseudo_inertia_constraint(pi, b=base):
            J = pseudo_inertia_matrix(pi[b:b + 10])
            return np.min(np.linalg.eigvalsh(J)) - eps_eig
        constraints.append({"type": "ineq", "fun": _pseudo_inertia_constraint})

    result = minimize(
        cost, pi0, jac=grad,
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )

    pi_hat_full = result.x
    res_norm = np.linalg.norm(W_full @ pi_hat_full - tau_vec)
    logger.info("Constrained LS (%s): residual=%.6e, success=%s, nit=%d",
                feasibility_method, res_norm, result.success, result.nit)
    if not result.success:
        logger.warning("Constrained optimisation did not converge: %s", result.message)

    return pi_hat_full, res_norm, {
        "method": feasibility_method,
        "success": result.success,
        "nit": result.nit,
        "message": result.message,
        "solved_in_full_space": True,
    }


# ── Cholesky reparameterisation helpers ────────────────────────────────

# Lower-triangular element ordering (row-major):
#   idx 0: L[0,0]  idx 1: L[1,0]  idx 2: L[1,1]  idx 3: L[2,0]
#   idx 4: L[2,1]  idx 5: L[2,2]  idx 6: L[3,0]  idx 7: L[3,1]
#   idx 8: L[3,2]  idx 9: L[3,3]
_LTRI_PAIRS = [(r, c) for r in range(4) for c in range(r + 1)]
_LTRI_DIAG_IDX = [i for i, (r, c) in enumerate(_LTRI_PAIRS) if r == c]


def _cholesky_vec_to_lower(L_vec):
    """10-element vector → 4×4 lower-triangular matrix."""
    L = np.zeros((4, 4))
    for idx, (r, c) in enumerate(_LTRI_PAIRS):
        L[r, c] = L_vec[idx]
    return L


def _lower_to_cholesky_vec(L):
    """4×4 lower-triangular matrix → 10-element vector."""
    return np.array([L[r, c] for r, c in _LTRI_PAIRS])


def _cholesky_pi_jacobian(L):
    """Compute the 10×10 Jacobian  d(pi_link) / d(L_vec).

    Chain:  L_vec → L → J = L L^T → pi_link = extract(J).
    """
    jac = np.zeros((10, 10))
    for idx, (r, c) in enumerate(_LTRI_PAIRS):
        # dJ/dL_{r,c}:  dJ[a,b] = δ(a,r)*L[b,c] + δ(b,r)*L[a,c]
        dJ = np.zeros((4, 4))
        dJ[r, :] = L[:, c]
        dJ[:, r] += L[:, c]
        # d(pi)/d(J) applied to dJ
        jac[0, idx] = dJ[3, 3]                  # m
        jac[1, idx] = dJ[0, 3]                  # hx (mx)
        jac[2, idx] = dJ[1, 3]                  # hy (my)
        jac[3, idx] = dJ[2, 3]                  # hz (mz)
        jac[4, idx] = dJ[1, 1] + dJ[2, 2]      # Ixx
        jac[5, idx] = -dJ[0, 1]                 # Ixy
        jac[6, idx] = -dJ[0, 2]                 # Ixz
        jac[7, idx] = dJ[0, 0] + dJ[2, 2]      # Iyy
        jac[8, idx] = -dJ[1, 2]                 # Iyz
        jac[9, idx] = dJ[0, 0] + dJ[1, 1]      # Izz
    return jac


def _solve_cholesky(W_base, tau_vec, nDoF, base_solver, weights, P_mat):
    """Solve LS in FULL parameter space with Cholesky reparameterisation.

    Each link's pseudo-inertia is parameterised as J = L L^T with L
    lower-triangular (10 free elements), guaranteeing J ≽ 0 by
    construction.  The optimisation is unconstrained in L-space.
    """
    p_full = P_mat.shape[1]
    W_full = W_base @ P_mat  # (m, p_full)
    n_rigid = min(nDoF, p_full // 10)
    n_extra = p_full - 10 * n_rigid

    # -- OLS initial guess on full space ------------------------------------
    pi0, _, _, _ = np.linalg.lstsq(W_full, tau_vec, rcond=None)

    # -- Convert OLS params → Cholesky factors (initial L_vec) --------------
    L_vec_all = np.zeros(10 * n_rigid)
    eps_init = 1e-8
    for i in range(n_rigid):
        b = 10 * i
        pi_link = pi0[b:b + 10]
        J = pseudo_inertia_matrix(pi_link)
        # Ensure PSD for Cholesky decomposition
        eigvals, eigvecs = np.linalg.eigh(J)
        eigvals_clipped = np.maximum(eigvals, eps_init)
        J_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        L = np.linalg.cholesky(J_psd)
        L_vec_all[b:b + 10] = _lower_to_cholesky_vec(L)

    # -- Build optimisation variable x = [L_vec_all, extra_params] ----------
    if n_extra > 0:
        x0 = np.concatenate([L_vec_all, pi0[10 * n_rigid:]])
    else:
        x0 = L_vec_all.copy()

    # -- Pre-compute for fast cost ------------------------------------------
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        W_eff = W_full * sqrt_w[:, None]
        tau_eff = tau_vec * sqrt_w
    else:
        W_eff = W_full
        tau_eff = tau_vec
    WtW = W_eff.T @ W_eff
    Wt_tau = W_eff.T @ tau_eff

    # -- Mapping x → pi_full -----------------------------------------------
    def _x_to_pi(x):
        pi = np.empty(p_full)
        for i in range(n_rigid):
            b = 10 * i
            L = _cholesky_vec_to_lower(x[b:b + 10])
            J = L @ L.T
            pi[b:b + 10] = pi_from_pseudo_inertia_matrix(J)
        if n_extra > 0:
            pi[10 * n_rigid:] = x[10 * n_rigid:]
        return pi

    # -- Cost + analytical gradient -----------------------------------------
    def cost_and_grad(x):
        pi = _x_to_pi(x)
        residual = W_eff @ pi - tau_eff
        c = 0.5 * np.dot(residual, residual)
        grad_pi = WtW @ pi - Wt_tau

        grad_x = np.empty_like(x)
        for i in range(n_rigid):
            b = 10 * i
            L = _cholesky_vec_to_lower(x[b:b + 10])
            jac_link = _cholesky_pi_jacobian(L)  # (10, 10)
            grad_x[b:b + 10] = jac_link.T @ grad_pi[b:b + 10]
        if n_extra > 0:
            grad_x[10 * n_rigid:] = grad_pi[10 * n_rigid:]
        return c, grad_x

    # -- Box bounds: diagonal elements of L >= eps --------------------------
    bounds_list = []
    for _ in range(n_rigid):
        for r, c in _LTRI_PAIRS:
            if r == c:
                bounds_list.append((1e-10, None))
            else:
                bounds_list.append((None, None))
    for _ in range(n_extra):
        bounds_list.append((None, None))

    result = minimize(
        cost_and_grad, x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_list,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
    )

    pi_hat_full = _x_to_pi(result.x)
    res_norm = np.linalg.norm(W_full @ pi_hat_full - tau_vec)
    logger.info("Cholesky-constrained LS: residual=%.6e, success=%s, nit=%d",
                res_norm, result.success, result.nit)
    if not result.success:
        logger.warning("Cholesky optimisation did not converge: %s", result.message)

    return pi_hat_full, res_norm, {
        "method": "cholesky",
        "success": result.success,
        "nit": result.nit,
        "message": result.message,
        "solved_in_full_space": True,
    }
