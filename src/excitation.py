"""Excitation trajectory optimisation with three constraint styles.

Styles
------
- "legacy_excTrajGen"    : heuristic cost, soft exp penalties, differential evolution
- "urdf_reference"       : minimize cond(Y^T Y), sigmoid constraints, diff. evolution
- "literature_standard"  : minimize cond(W_base^T W_base), hard inequality constraints,
                           SLSQP with base-parameter regressor (Swevers et al. 1997)
"""
import logging
import numpy as np
from scipy.optimize import differential_evolution, minimize

from .trajectory import (fourier_trajectory, build_frequencies,
                          param_count, param_bounds)
from .dynamics_newton_euler import newton_euler_regressor
from .friction import build_friction_regressor
from .base_parameters import compute_base_parameters

logger = logging.getLogger("sysid_pipeline")


def optimise_excitation(kin, cfg_exc, q_lim, dq_lim, ddq_lim,
                        friction_model="none", regressor_fn=None):
    """Run excitation-trajectory optimisation. Returns optimal trajectory params.

    Parameters
    ----------
    kin : RobotKinematics
    cfg_exc : excitation section of configuration
    q_lim, dq_lim, ddq_lim : (nDoF, 2) limit arrays
    friction_model : friction model string
    regressor_fn : callable for EL regressor (or None -> use NE)

    Returns
    -------
    result : dict with keys "params", "freqs", "q0", "cost", "traj_cfg"
    """
    nDoF = kin.nDoF
    basis = cfg_exc["basis_functions"]
    opt_phase = cfg_exc.get("optimize_phase", False)
    m = cfg_exc["num_harmonics"]
    f0 = cfg_exc["base_frequency_hz"]
    style = cfg_exc["constraint_style"]
    max_iter = cfg_exc.get("optimizer_max_iter", 300)
    pop_size = cfg_exc.get("optimizer_pop_size", 15)
    n_periods = cfg_exc.get("trajectory_duration_periods", 1)
    opt_cond = cfg_exc.get("optimize_condition_number", True)

    freqs = build_frequencies(f0, m)
    T_period = 1.0 / f0
    tf = n_periods * T_period
    dt_nyquist = 1.0 / (2.0 * freqs[-1])
    t = np.arange(0, tf + dt_nyquist, dt_nyquist)
    q0 = np.mean(q_lim, axis=1)

    n_params = param_count(nDoF, m, basis, opt_phase)
    bounds = param_bounds(nDoF, m, basis, opt_phase, q_lim)

    logger.info("Excitation optimisation: style=%s, basis=%s, harmonics=%d, "
                "optimize_cond=%s, n_params=%d",
                style, basis, m, opt_cond, n_params)

    def _get_regressor(q_val, dq_val, ddq_val):
        if regressor_fn is not None:
            Y = regressor_fn(q_val, dq_val, ddq_val)
        else:
            Y = newton_euler_regressor(kin, q_val, dq_val, ddq_val)
        if friction_model != "none":
            Yf = build_friction_regressor(dq_val, friction_model)
            Y = np.hstack((Y, Yf))
        return Y

    # -- Style: legacy_excTrajGen -----------------------------------------------
    if style == "legacy_excTrajGen":
        def cost(x):
            q, dq, _ = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
            j = np.sum(1.0 / (1e-3 + np.abs(q))) + 100.0 * np.sum(1.0 / (1e-3 + np.abs(dq)))
            return j + _penalty_soft_exp(q, dq, q_lim, dq_lim)

        result = _run_differential_evolution(cost, bounds, max_iter, pop_size)

    # -- Style: urdf_reference --------------------------------------------------
    elif style == "urdf_reference":
        def cost(x):
            q, dq, ddq_v = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
            if opt_cond:
                c = _condition_cost(q, dq, ddq_v, t, kin, _get_regressor)
            else:
                c = _amplitude_cost(dq, ddq_v)
            return c + _penalty_sigmoid(q, dq, ddq_v, q_lim, dq_lim, ddq_lim)

        result = _run_differential_evolution(cost, bounds, max_iter, pop_size)

    # -- Style: literature_standard ---------------------------------------------
    elif style == "literature_standard":
        def cost_lit(x):
            q, dq, ddq_v = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
            if opt_cond:
                c = _condition_cost_base(q, dq, ddq_v, t, kin, _get_regressor, nDoF)
            else:
                c = _amplitude_cost(dq, ddq_v)
            return c

        constraints = _build_slsqp_constraints(
            freqs, t, q0, basis, opt_phase, q_lim, dq_lim, ddq_lim, nDoF
        )

        # Initial guess: small random within bounds
        rng = np.random.default_rng(42)
        x0 = np.array([(lo + hi) / 2.0 + (hi - lo) * 0.1 * rng.standard_normal()
                        for lo, hi in bounds])
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

        res = minimize(
            cost_lit, x0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False, "ftol": 1e-8},
        )
        if not res.success:
            logger.warning("SLSQP excitation optimisation did NOT converge: %s "
                           "(nit=%d, cost=%.6f). Consider increasing max_iter or "
                           "adjusting bounds.", res.message, res.nit, res.fun)
        else:
            logger.info("SLSQP finished: cost=%.6f, success=%s, nit=%d",
                        res.fun, res.success, res.nit)

        # Post-optimisation constraint verification
        q_check, dq_check, ddq_check = fourier_trajectory(
            res.x, freqs, t, q0, basis, opt_phase
        )
        violations = []
        for i in range(nDoF):
            if np.max(q_check[i]) > q_lim[i, 1] + 1e-6:
                violations.append(f"q[{i}] max={np.max(q_check[i]):.4f} > {q_lim[i,1]:.4f}")
            if np.min(q_check[i]) < q_lim[i, 0] - 1e-6:
                violations.append(f"q[{i}] min={np.min(q_check[i]):.4f} < {q_lim[i,0]:.4f}")
            if np.max(dq_check[i]) > dq_lim[i, 1] + 1e-6:
                violations.append(f"dq[{i}] max={np.max(dq_check[i]):.4f} > {dq_lim[i,1]:.4f}")
            if np.min(dq_check[i]) < dq_lim[i, 0] - 1e-6:
                violations.append(f"dq[{i}] min={np.min(dq_check[i]):.4f} < {dq_lim[i,0]:.4f}")
            if np.max(ddq_check[i]) > ddq_lim[i, 1] + 1e-6:
                violations.append(f"ddq[{i}] max={np.max(ddq_check[i]):.4f} > {ddq_lim[i,1]:.4f}")
            if np.min(ddq_check[i]) < ddq_lim[i, 0] - 1e-6:
                violations.append(f"ddq[{i}] min={np.min(ddq_check[i]):.4f} < {ddq_lim[i,0]:.4f}")
        if violations:
            logger.warning("Post-optimisation constraint violations: %s",
                           "; ".join(violations))

        result = res
    else:
        raise ValueError(f"Unknown constraint style: {style}")

    logger.info("Excitation optimisation done. cost=%.6f", result.fun)

    return {
        "params": result.x,
        "freqs": freqs,
        "q0": q0,
        "cost": result.fun,
        "basis": basis,
        "optimize_phase": opt_phase,
    }


# -- Solver wrappers --------------------------------------------------------

def _run_differential_evolution(cost, bounds, max_iter, pop_size):
    return differential_evolution(
        cost, bounds,
        strategy="best1bin",
        maxiter=max_iter,
        popsize=pop_size,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False,
        seed=42,
    )


# -- Cost functions ----------------------------------------------------------

def _condition_cost(q, dq, ddq, t, kin, get_reg_fn):
    """Condition number on the full (unreduced) stacked regressor."""
    N = t.size
    step = max(1, N // 50)
    indices = list(range(0, N, step))
    rows = [get_reg_fn(q[:, idx], dq[:, idx], ddq[:, idx]) for idx in indices]
    W = np.vstack(rows)
    return _cond_from_matrix(W)


def _condition_cost_base(q, dq, ddq, t, kin, get_reg_fn, nDoF):
    """Condition number on the BASE-PARAMETER observation matrix.

    Builds W, reduces via QR, then returns cond(W_base).
    """
    N = t.size
    step = max(1, N // 50)
    indices = list(range(0, N, step))
    rows = [get_reg_fn(q[:, idx], dq[:, idx], ddq[:, idx]) for idx in indices]
    W = np.vstack(rows)

    # Reduce to base parameters
    p = W.shape[1]
    pi_dummy = np.ones(p)
    try:
        W_base, _, _, rank, _ = compute_base_parameters(W, pi_dummy, tol=1e-8)
    except ValueError:
        return 1e12
    return _cond_from_matrix(W_base)


def _cond_from_matrix(W):
    try:
        sv = np.linalg.svd(W, compute_uv=False)
        sv_pos = sv[sv > 1e-12]
        if len(sv_pos) < 2:
            return 1e12
        return sv_pos[0] / sv_pos[-1]
    except np.linalg.LinAlgError:
        return 1e12


def _amplitude_cost(dq, ddq):
    """When not optimising condition number: maximise velocity/accel amplitude."""
    return -(np.sum(np.abs(dq)) + np.sum(np.abs(ddq)))


# -- Constraint builders for SLSQP ------------------------------------------

def _build_slsqp_constraints(freqs, t, q0, basis, opt_phase,
                              q_lim, dq_lim, ddq_lim, nDoF):
    """Build hard inequality constraints for scipy.optimize.minimize(SLSQP).

    Each constraint is  c(x) >= 0.
    """
    cons = []

    def _make_upper(idx, limit_val, signal_type):
        """limit_val - signal[idx, :] >= 0  (upper bound)."""
        def con(x):
            q, dq, ddq = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
            sig = {"q": q, "dq": dq, "ddq": ddq}[signal_type]
            return limit_val - np.max(sig[idx])
        return con

    def _make_lower(idx, limit_val, signal_type):
        """signal[idx, :] - limit_val >= 0  (lower bound)."""
        def con(x):
            q, dq, ddq = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
            sig = {"q": q, "dq": dq, "ddq": ddq}[signal_type]
            return np.min(sig[idx]) - limit_val
        return con

    for i in range(nDoF):
        cons.append({"type": "ineq", "fun": _make_upper(i, q_lim[i, 1], "q")})
        cons.append({"type": "ineq", "fun": _make_lower(i, q_lim[i, 0], "q")})
        cons.append({"type": "ineq", "fun": _make_upper(i, dq_lim[i, 1], "dq")})
        cons.append({"type": "ineq", "fun": _make_lower(i, dq_lim[i, 0], "dq")})
        cons.append({"type": "ineq", "fun": _make_upper(i, ddq_lim[i, 1], "ddq")})
        cons.append({"type": "ineq", "fun": _make_lower(i, ddq_lim[i, 0], "ddq")})

    return cons


# -- Penalty functions -------------------------------------------------------

def _penalty_soft_exp(q, dq, q_lim, dq_lim, alpha=100.0, weight=1e2):
    """Legacy soft exponential penalties."""
    c = 0.0
    nDoF = q.shape[0]
    for i in range(nDoF):
        rng = q_lim[i, 1] - q_lim[i, 0]
        c += np.sum(1.0 / np.exp(alpha * (q_lim[i, 1] - q[i]) / rng)
                    + 1.0 / np.exp(alpha * (q[i] - q_lim[i, 0]) / rng)
                    - 2.0 / np.exp(alpha / 2.0))
        rng_v = dq_lim[i, 1] - dq_lim[i, 0]
        c += np.sum(1.0 / np.exp(alpha * (dq_lim[i, 1] - dq[i]) / rng_v)
                    + 1.0 / np.exp(alpha * (dq[i] - dq_lim[i, 0]) / rng_v)
                    - 2.0 / np.exp(alpha / 2.0))
    return weight * c


def _penalty_sigmoid(q, dq, ddq, q_lim, dq_lim, ddq_lim,
                     alpha_q=50.0, alpha_dq=50.0, alpha_ddq=50.0, weight=1e3):
    """Sigmoid two-way inequality constraints (Ref-2 S4.4)."""
    def sigmoid_pen(x, lo, hi, alpha):
        return np.sum(2.0 / (1.0 + np.exp(alpha * (hi - x)))
                      - 2.0 / (1.0 + np.exp(-alpha * (lo - x))))
    c = 0.0
    for i in range(q.shape[0]):
        c += sigmoid_pen(q[i], q_lim[i, 0], q_lim[i, 1], alpha_q)
        c += sigmoid_pen(dq[i], dq_lim[i, 0], dq_lim[i, 1], alpha_dq)
        c += sigmoid_pen(ddq[i], ddq_lim[i, 0], ddq_lim[i, 1], alpha_ddq)
    return weight * max(0.0, c)
