"""Parameter identification solvers: OLS, WLS, bounded least-squares,
and physically-constrained least-squares.

The constrained path ("lmi") enforces pseudo-inertia PSD per link, which
is the correct necessary-and-sufficient condition for physical consistency
(see Wensing, Kim & Slotine 2018).  The solver wraps the LS problem inside
an SLSQP optimisation with per-link eigenvalue constraints on the 4×4
pseudo-inertia matrix J.

Constrained identification requires method="newton_euler" so that the
full 10-per-link parameter structure is preserved.  The config validator
rejects euler_lagrange + feasibility != "none".

Note: "cholesky" is accepted as a deprecated alias for "lmi" by the config
loader.  No true Cholesky-factored reparameterisation is implemented.
"""
import logging
import numpy as np
from scipy.optimize import lsq_linear, minimize

from .feasibility import pseudo_inertia_matrix

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
    feasibility_method : "none" | "lmi"
        When "lmi", the solver wraps the LS problem inside an SLSQP
        optimisation with physical feasibility constraints on the FULL
        parameter vector (10 per link).  Requires method="newton_euler".
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
    if feasibility_method in ("lmi", "cholesky") and nDoF > 0:
        if P_mat is None:
            logger.warning("Constrained identification requested but no regrouping "
                           "matrix P provided. Falling back to unconstrained solver.")
        else:
            return _solve_constrained(W, tau_vec, nDoF, feasibility_method,
                                      solver, weights, P_mat)

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
