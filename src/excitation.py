"""Excitation trajectory optimisation with torque-limit-aware extensions."""
# Long-horizon sine-only excitation is fundamentally drift-limited: the
# lambda_1 correction required to enforce dq(0)=0 bounds all sine amplitudes by
# O(q_range / tf), so the feasible set collapses toward near-zero motion as tf
# grows. Use basis="both" with optimize_phase=False for long trajectories.
import logging
from copy import deepcopy

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .base_parameters import compute_base_parameters
from .dynamics_newton_euler import newton_euler_regressor
from .torque_constraints import (
    HARD_TORQUE_METHODS,
    compute_soft_penalty,
    compute_torque_design_data,
    make_augmented_regressor,
    validation_time_vector,
)
from .trajectory import (
    build_frequencies,
    fourier_trajectory,
    param_bounds,
    param_count,
)

logger = logging.getLogger("sysid_pipeline")
_SINE_USABLE_FRACTION_MIN = 0.01


def _symmetric_limit(limit_row):
    """Return the smaller-magnitude side of a symmetric-ish limit interval."""
    return float(min(abs(limit_row[0]), abs(limit_row[1])))


def _lam1_bound(q_limit_row, dq_limit_row, tf):
    """Maximum admissible lambda_1 drift magnitude for one DOF."""
    q_range = float(q_limit_row[1] - q_limit_row[0])
    return min(_symmetric_limit(dq_limit_row), q_range / (2.0 * tf))


def _trajectory_violates_limits(q, dq, ddq, q_lim, dq_lim, ddq_lim, tol=1e-9):
    """Return True when any sampled trajectory point violates signal limits."""
    return (
        np.any(q > q_lim[:, 1:2] + tol)
        or np.any(q < q_lim[:, 0:1] - tol)
        or np.any(dq > dq_lim[:, 1:2] + tol)
        or np.any(dq < dq_lim[:, 0:1] - tol)
        or np.any(ddq > ddq_lim[:, 1:2] + tol)
        or np.any(ddq < ddq_lim[:, 0:1] - tol)
    )


def _sine_basis_infeasibility_warning(tf, usable_fraction):
    """Return the standard warning for drift-limited sine trajectories."""
    return (
        "basis='sine' with tf=%.1f s constrains all amplitudes to <=%.4f%% "
        "of their individual bounds due to the lambda_1 drift cap. The "
        "trajectory will be nearly flat; switch to basis='both' with "
        "optimize_phase=false for long trajectories."
    ) % (tf, usable_fraction * 100.0)


def _sine_basis_infeasibility_error(tf, usable_fraction):
    """Return the standard error for fundamentally infeasible sine trajectories."""
    return (
        "basis='sine' is fundamentally infeasible for this trajectory "
        f"duration (tf={tf:.1f} s): the lambda_1 drift cap limits usable "
        f"amplitudes to <= {usable_fraction * 100.0:.4f}% of their "
        "per-harmonic bounds. Switch to basis='both' with "
        "optimize_phase=false."
    )


def _sine_basis_usable_fraction(bounds, basis, nDoF, m, freqs, q_lim, dq_lim, tf):
    """Return the smallest usable sine-amplitude fraction after lambda_1 limits."""
    if basis != "sine":
        return None

    usable_fractions = []
    twopi = 2.0 * np.pi
    for i in range(nDoF):
        b_slice = bounds[i * m: (i + 1) * m]
        amp_bounds = np.array([hi for _, hi in b_slice], dtype=float)
        amp_f_sum = float(np.dot(amp_bounds, freqs))
        lam1_max = _lam1_bound(q_lim[i], dq_lim[i], tf)
        usable_fractions.append(lam1_max / max(twopi * amp_f_sum, 1e-12))

    usable_fraction = min(usable_fractions, default=1.0)
    return usable_fraction


def _validate_sine_basis_feasibility(bounds, basis, nDoF, m, freqs, q_lim, dq_lim, tf,
                                     logger_=None):
    """Warn and raise when sine-only excitation is drift-limited to near-zero motion."""
    usable_fraction = _sine_basis_usable_fraction(
        bounds, basis, nDoF, m, freqs, q_lim, dq_lim, tf,
    )
    if usable_fraction is not None and usable_fraction < _SINE_USABLE_FRACTION_MIN:
        if logger_ is not None:
            logger_.warning(_sine_basis_infeasibility_warning(tf, usable_fraction))
        raise ValueError(_sine_basis_infeasibility_error(tf, usable_fraction))
    return usable_fraction


def preflight_excitation_config(cfg_exc, q_lim, dq_lim, ddq_lim, logger_=None):
    """Validate excitation settings that depend on extracted joint limits."""
    basis = cfg_exc["basis_functions"]
    opt_phase = cfg_exc.get("optimize_phase", False)
    n_dof = q_lim.shape[0]
    m = cfg_exc["num_harmonics"]
    f0 = cfg_exc["base_frequency_hz"]
    n_periods = cfg_exc.get("trajectory_duration_periods", 1)

    freqs = build_frequencies(f0, m)
    tf = n_periods / f0
    bounds = param_bounds(
        n_dof, m, basis, opt_phase, q_lim,
        freqs=freqs, dq_lim=dq_lim, ddq_lim=ddq_lim, tf=tf,
    )
    usable_fraction = _validate_sine_basis_feasibility(
        bounds, basis, n_dof, m, freqs, q_lim, dq_lim, tf, logger_,
    )
    return {
        "basis": basis,
        "optimize_phase": opt_phase,
        "n_dof": n_dof,
        "num_harmonics": m,
        "freqs": freqs,
        "tf": tf,
        "bounds": bounds,
        "usable_fraction": usable_fraction,
    }


def _build_literature_initial_guess(bounds, basis, opt_phase, nDoF, m, freqs,
                                    q0, q_lim, dq_lim, ddq_lim, t, tf, rng):
    """Construct a trajectory-aware initial guess for the literature solver."""
    if basis == "both" and not opt_phase:
        a_scale = 0.5 / m
        x0_list = []
        for i in range(nDoF):
            a_slice = bounds[i * 2 * m: i * 2 * m + m]
            b_slice = bounds[i * 2 * m + m: i * 2 * m + 2 * m]
            # Alternate signs across harmonics so that the lam0 = -sum(a_j)
            # contributions partially cancel, reducing baseline offset
            # bias.  This is a heuristic that gives the optimizer a better
            # starting point; the first harmonic still dominates so the
            # initial trajectory may remain somewhat one-sided.
            for j, (_, hi) in enumerate(a_slice):
                sign = 1.0 if j % 2 == 0 else -1.0
                x0_list.append(sign * hi * a_scale * (1.0 + 0.05 * rng.standard_normal()))
            # Use the same a_scale for sine amplitudes (instead of the
            # previous 0.001) since the bounds are now correctly
            # tightened for the secular drift (Fix 1).  Alternate signs
            # for richer initial velocity profiles.
            for j, (_, hi) in enumerate(b_slice):
                sign = 1.0 if j % 2 == 0 else -1.0
                x0_list.append(sign * hi * a_scale * (1.0 + 0.05 * rng.standard_normal()))
        x0 = np.array(x0_list)
    elif basis == "sine":
        x0_list = []
        twopi = 2.0 * np.pi
        omega = twopi * freqs
        for i in range(nDoF):
            b_slice = bounds[i * m: (i + 1) * m]
            amp_bounds = np.array([hi for _, hi in b_slice], dtype=float)
            harmonic_q_cap = (q_lim[i, 1] - q_lim[i, 0]) / max(2.0 * np.sum(amp_bounds), 1e-12)
            harmonic_dq_cap = _symmetric_limit(dq_lim[i]) / max(np.sum(amp_bounds * omega), 1e-12)
            harmonic_ddq_cap = _symmetric_limit(ddq_lim[i]) / max(np.sum(amp_bounds * omega ** 2), 1e-12)
            lam1_cap = _lam1_bound(q_lim[i], dq_lim[i], tf) / max(twopi * np.dot(amp_bounds, freqs), 1e-12)
            safe_scale = 0.9 * min(0.1, harmonic_q_cap, harmonic_dq_cap, harmonic_ddq_cap, lam1_cap)
            for _, hi in b_slice:
                x0_list.append(hi * safe_scale * (1.0 + 0.05 * rng.standard_normal()))
        x0 = np.array(x0_list)

        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        x0 = np.clip(x0, lower, upper)
        for _ in range(20):
            q, dq, ddq = fourier_trajectory(x0, freqs, t, q0, basis, opt_phase)
            if not _trajectory_violates_limits(q, dq, ddq, q_lim, dq_lim, ddq_lim):
                break
            x0 *= 0.5
    else:
        x0 = np.array([
            hi * 0.1 * (1.0 + 0.05 * rng.standard_normal())
            for _, hi in bounds
        ])

    return np.clip(
        x0,
        np.array([b[0] for b in bounds], dtype=float),
        np.array([b[1] for b in bounds], dtype=float),
    )


def optimise_excitation(kin, cfg_exc, q_lim, dq_lim, ddq_lim,
                        friction_model="none", regressor_fn=None,
                        tau_lim=None, nominal_params=None):
    """Run excitation-trajectory optimisation and return optimal trajectory params."""
    nDoF = kin.nDoF
    basis = cfg_exc["basis_functions"]
    opt_phase = cfg_exc.get("optimize_phase", False)
    m = cfg_exc["num_harmonics"]
    f0 = cfg_exc["base_frequency_hz"]
    max_iter = cfg_exc.get("optimizer_max_iter", 300)
    n_periods = cfg_exc.get("trajectory_duration_periods", 1)
    opt_cond = cfg_exc.get("optimize_condition_number", True)
    torque_method = cfg_exc.get("torque_constraint_method", "none")
    torque_cfg = deepcopy(cfg_exc.get("torque_constraint", {}))
    oversample = cfg_exc.get("torque_validation_oversample_factor", 1)

    freqs = build_frequencies(f0, m)
    T_period = 1.0 / f0
    tf = n_periods * T_period
    dt_nyquist = 1.0 / (2.0 * freqs[-1])
    t = np.arange(0.0, tf + dt_nyquist, dt_nyquist)
    q0 = np.mean(q_lim, axis=1)

    n_params = param_count(nDoF, m, basis, opt_phase)
    bounds = param_bounds(
        nDoF, m, basis, opt_phase, q_lim,
        freqs=freqs, dq_lim=dq_lim, ddq_lim=ddq_lim, tf=tf,
    )
    _validate_sine_basis_feasibility(
        bounds, basis, nDoF, m, freqs, q_lim, dq_lim, tf, logger,
    )

    logger.info(
        "Excitation optimisation: basis=%s, harmonics=%d, optimize_cond=%s, "
        "n_params=%d, torque_method=%s",
        basis, m, opt_cond, n_params, torque_method,
    )

    if regressor_fn is not None:
        base_regressor_fn = regressor_fn
    else:
        def base_regressor_fn(q_val, dq_val, ddq_val):
            return newton_euler_regressor(kin, q_val, dq_val, ddq_val)

    get_regressor = make_augmented_regressor(base_regressor_fn, friction_model)

    # Structural identifiability depends only on the robot model, not on the
    # specific trajectory. Reusing kept_cols avoids an expensive QR decomposition
    # on every cost/gradient evaluation when optimizing the base-matrix condition
    # number.
    base_kept_cols = None
    if opt_cond:
        try:
            q_ref = np.tile(q0[:, None], (1, 50)) + 0.1 * np.random.default_rng(0).standard_normal((nDoF, 50))
            dq_ref = 0.5 * np.random.default_rng(1).standard_normal((nDoF, 50))
            ddq_ref = 0.5 * np.random.default_rng(2).standard_normal((nDoF, 50))
            W_ref = np.vstack([
                get_regressor(q_ref[:, k], dq_ref[:, k], ddq_ref[:, k])
                for k in range(50)
            ])
            _, _, base_kept_cols, _, _ = compute_base_parameters(
                W_ref, np.ones(W_ref.shape[1])
            )
        except Exception:
            base_kept_cols = None

    def cost_lit(x):
        q, dq, ddq_v = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
        if opt_cond:
            if base_kept_cols is not None:
                c_raw = _condition_cost_base_fast(q, dq, ddq_v, t, get_regressor, base_kept_cols)
            else:
                c_raw = _condition_cost_base(q, dq, ddq_v, t, get_regressor)
            # Use log10(cond) instead of raw cond to avoid gradient scale
            # mismatch with constraint gradients.  The raw condition number
            # can be O(10^4-10^5) at small amplitudes, producing gradients
            # of O(10^8) vs constraint gradients of O(10^2).  log10 is
            # monotone so the optimum is unchanged, but the gradient is
            # reduced by a factor of ~cond*ln(10), matching constraint
            # scales and preventing SLSQP QP infeasibility.
            c = np.log10(max(c_raw, 1.0))
        else:
            c = _amplitude_cost(dq, ddq_v)

        if torque_method == "soft_penalty" and tau_lim is not None and nominal_params is not None:
            design = compute_torque_design_data(
                q, dq, ddq_v, get_regressor, nominal_params, tau_lim,
                "nominal_hard", torque_cfg,
            )
            c += compute_soft_penalty(
                design["tau_nominal"],
                design["limit_lower"],
                design["limit_upper"],
                torque_cfg.get("soft_penalty_weight", 100.0),
                torque_cfg.get("soft_penalty_smoothing", 0.01),
            )
        return c

    # Using the same oversampled grid as the dense torque validation ensures
    # that optimization feasibility implies dense-validation feasibility.
    dt_con = dt_nyquist / max(1, oversample)
    t_con = np.arange(0.0, tf + dt_con, dt_con)

    if basis == "sine" or (basis == "both" and not opt_phase):
        constraints = _build_linear_traj_constraints(
            freqs, t_con, q0, q_lim, dq_lim, ddq_lim, nDoF, m, basis=basis
        )
    else:
        constraints = _build_slsqp_constraints(
            freqs, t, q0, basis, opt_phase, q_lim, dq_lim, ddq_lim, nDoF,
            m=m, tf=tf,
        )
    constraints += _build_torque_constraints(
        freqs, t, q0, basis, opt_phase, nDoF,
        torque_method=torque_method,
        tau_lim=tau_lim,
        nominal_params=nominal_params,
        get_regressor_fn=get_regressor,
        torque_cfg=torque_cfg,
    )

    rng = np.random.default_rng(42)
    x0 = _build_literature_initial_guess(
        bounds, basis, opt_phase, nDoF, m, freqs, q0, q_lim, dq_lim,
        ddq_lim, t, tf, rng,
    )
    result = minimize(
        cost_lit,
        x0,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-8},
    )
    if not result.success:
        logger.warning(
            "SLSQP excitation optimisation did NOT converge: %s (nit=%d, cost=%.6f). "
            "Consider increasing max_iter or adjusting bounds.",
            result.message, result.nit, result.fun,
        )
    else:
        logger.info("SLSQP finished: cost=%.6f, success=%s, nit=%d",
                    result.fun, result.success, result.nit)

    q_check, dq_check, ddq_check = fourier_trajectory(result.x, freqs, t, q0, basis, opt_phase)
    violations = []
    for i in range(nDoF):
        if np.max(q_check[i]) > q_lim[i, 1] + 1e-6:
            violations.append(f"q[{i}] max={np.max(q_check[i]):.4f} > {q_lim[i, 1]:.4f}")
        if np.min(q_check[i]) < q_lim[i, 0] - 1e-6:
            violations.append(f"q[{i}] min={np.min(q_check[i]):.4f} < {q_lim[i, 0]:.4f}")
        if np.max(dq_check[i]) > dq_lim[i, 1] + 1e-6:
            violations.append(f"dq[{i}] max={np.max(dq_check[i]):.4f} > {dq_lim[i, 1]:.4f}")
        if np.min(dq_check[i]) < dq_lim[i, 0] - 1e-6:
            violations.append(f"dq[{i}] min={np.min(dq_check[i]):.4f} < {dq_lim[i, 0]:.4f}")
        if np.max(ddq_check[i]) > ddq_lim[i, 1] + 1e-6:
            violations.append(f"ddq[{i}] max={np.max(ddq_check[i]):.4f} > {ddq_lim[i, 1]:.4f}")
        if np.min(ddq_check[i]) < ddq_lim[i, 0] - 1e-6:
            violations.append(f"ddq[{i}] min={np.min(ddq_check[i]):.4f} < {ddq_lim[i, 0]:.4f}")
    if violations:
        logger.warning("Post-optimisation constraint violations: %s", "; ".join(violations))

    if torque_method in HARD_TORQUE_METHODS and tau_lim is not None and nominal_params is not None:
        t_dense = validation_time_vector(freqs, f0, n_periods, oversample)
        q_dense, dq_dense, ddq_dense = fourier_trajectory(
            result.x, freqs, t_dense, q0, basis, opt_phase
        )
        design = compute_torque_design_data(
            q_dense, dq_dense, ddq_dense, get_regressor, nominal_params,
            tau_lim, torque_method, torque_cfg,
        )
        if not design["design_pass"]:
            message = f"Dense torque replay violated {torque_method} limits after optimization."
            if torque_cfg.get("strict_validation", True):
                raise ValueError(message)
            logger.warning(message)

    logger.info("Excitation optimisation done. cost=%.6f", result.fun)
    return {
        "params": result.x,
        "freqs": freqs,
        "q0": q0,
        "cost": result.fun,
        "basis": basis,
        "optimize_phase": opt_phase,
        "torque_constraint_method": torque_method,
    }


def _condition_cost_base(q, dq, ddq, t, get_reg_fn):
    """Condition number on the base-parameter observation matrix."""
    N = t.size
    step = max(1, N // 50)
    indices = list(range(0, N, step))
    rows = [get_reg_fn(q[:, idx], dq[:, idx], ddq[:, idx]) for idx in indices]
    W = np.vstack(rows)

    p = W.shape[1]
    pi_dummy = np.ones(p)
    try:
        W_base, _, _, _, _ = compute_base_parameters(W, pi_dummy, tol=1e-8)
    except ValueError:
        return 1e12
    return _cond_from_matrix(W_base)


def _condition_cost_base_fast(q, dq, ddq, t, get_reg_fn, kept_cols):
    """Condition number using pre-known base parameter column indices."""
    N = t.size
    step = max(1, N // 50)
    indices = list(range(0, N, step))
    rows = [get_reg_fn(q[:, idx], dq[:, idx], ddq[:, idx]) for idx in indices]
    W = np.vstack(rows)
    return _cond_from_matrix(W[:, kept_cols])


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


def _build_linear_traj_constraints(freqs, t_con, q0, q_lim, dq_lim, ddq_lim,
                                   nDoF, m, basis="both"):
    """Pre-build LinearConstraint objects for q/dq/ddq trajectory bounds.

    The Fourier trajectory is linear in x for:
      - basis="sine"
      - basis="both" with optimize_phase=False

    In both cases:
      q_i(t)   = A_q_i   @ x  + q0[i]
      dq_i(t)  = A_dq_i  @ x
      ddq_i(t) = A_ddq_i @ x

    Each row of A corresponds to one time point. scipy processes LinearConstraint
    objects analytically in the QP subproblem: no finite-difference Jacobians,
    no non-smooth argmax/argmin kinks that cause "Positive directional derivative"
    linesearch failures.

    Using the same oversampled grid as the dense torque validation ensures
    optimisation feasibility implies dense-validation feasibility.
    """
    if basis not in {"sine", "both"}:
        raise ValueError(f"Linear trajectory constraints do not support basis={basis!r}")

    N = len(t_con)
    n_params = nDoF * m if basis == "sine" else nDoF * 2 * m
    twopi_f = 2.0 * np.pi * freqs

    phase = np.outer(twopi_f, t_con)
    cos_h = np.cos(phase)
    sin_h = np.sin(phase)

    J_q_a = (cos_h - 1.0).T
    J_q_b = (sin_h - twopi_f[:, None] * t_con).T
    J_dq_a = (-twopi_f[:, None] * sin_h).T
    J_dq_b = (twopi_f[:, None] * (cos_h - 1.0)).T
    J_ddq_a = (-(twopi_f ** 2)[:, None] * cos_h).T
    J_ddq_b = (-(twopi_f ** 2)[:, None] * sin_h).T

    cons = []
    for i in range(nDoF):
        A_q = np.zeros((N, n_params))
        A_dq = np.zeros((N, n_params))
        A_ddq = np.zeros((N, n_params))
        if basis == "sine":
            off = i * m
            A_q[:, off:off + m] = J_q_b
            A_dq[:, off:off + m] = J_dq_b
            A_ddq[:, off:off + m] = J_ddq_b
        else:
            off = i * 2 * m
            A_q[:, off:off + m] = J_q_a
            A_q[:, off + m:off + 2 * m] = J_q_b
            A_dq[:, off:off + m] = J_dq_a
            A_dq[:, off + m:off + 2 * m] = J_dq_b
            A_ddq[:, off:off + m] = J_ddq_a
            A_ddq[:, off + m:off + 2 * m] = J_ddq_b

        cons.append(LinearConstraint(A_q,
                                     lb=q_lim[i, 0] - q0[i],
                                     ub=q_lim[i, 1] - q0[i]))
        cons.append(LinearConstraint(A_dq, lb=dq_lim[i, 0], ub=dq_lim[i, 1]))
        cons.append(LinearConstraint(A_ddq, lb=ddq_lim[i, 0], ub=ddq_lim[i, 1]))

    return cons


def _build_slsqp_constraints(freqs, t, q0, basis, opt_phase,
                             q_lim, dq_lim, ddq_lim, nDoF,
                             m=None, tf=None):
    """Fallback max/min trajectory constraints for non-linear-in-x bases."""
    cons = []
    _cx = [None]
    _ct = [None]

    def _get_traj(x):
        if _cx[0] is None or not np.array_equal(x, _cx[0]):
            _cx[0] = x.copy()
            _ct[0] = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
        return _ct[0]

    def _make_upper(idx, limit_val, signal_type):
        def con(x):
            q, dq, ddq = _get_traj(x)
            return limit_val - np.max({"q": q, "dq": dq, "ddq": ddq}[signal_type][idx])
        return {"type": "ineq", "fun": con}

    def _make_lower(idx, limit_val, signal_type):
        def con(x):
            q, dq, ddq = _get_traj(x)
            return np.min({"q": q, "dq": dq, "ddq": ddq}[signal_type][idx]) - limit_val
        return {"type": "ineq", "fun": con}

    for i in range(nDoF):
        cons.append(_make_upper(i, q_lim[i, 1], "q"))
        cons.append(_make_lower(i, q_lim[i, 0], "q"))
        cons.append(_make_upper(i, dq_lim[i, 1], "dq"))
        cons.append(_make_lower(i, dq_lim[i, 0], "dq"))
        cons.append(_make_upper(i, ddq_lim[i, 1], "ddq"))
        cons.append(_make_lower(i, ddq_lim[i, 0], "ddq"))

    if m is not None and tf is not None and (basis == "sine" or (basis == "both" and not opt_phase)):
        twopi = 2.0 * np.pi
        if basis == "sine":
            n_p = nDoF * m

            def _b_off(dof):
                return dof * m
        else:
            n_p = nDoF * 2 * m

            def _b_off(dof):
                return dof * 2 * m + m

        def _make_lam1(dof, bound, sign):
            off = _b_off(dof)
            _g = np.zeros(n_p)
            _g[off: off + m] = sign * twopi * freqs

            def con(x):
                b_i = x[off: off + m]
                lam1 = -twopi * np.dot(b_i, freqs)
                return bound - sign * lam1

            return {"type": "ineq", "fun": con, "jac": lambda x, _g=_g: _g}

        for i in range(nDoF):
            lam1_max = _lam1_bound(q_lim[i], dq_lim[i], tf)
            cons.append(_make_lam1(i, lam1_max, +1))
            cons.append(_make_lam1(i, lam1_max, -1))

    return cons


def _build_torque_constraints(freqs, t, q0, basis, opt_phase, nDoF,
                              torque_method="none", tau_lim=None,
                              nominal_params=None, get_regressor_fn=None,
                              torque_cfg=None):
    """Nonlinear torque constraints (shared trajectory cache)."""
    if torque_method not in HARD_TORQUE_METHODS or tau_lim is None \
            or nominal_params is None or get_regressor_fn is None:
        return []

    torque_cfg = torque_cfg or {}
    # Tighten design-time torque bounds slightly to absorb inter-sample peaks
    # observed by the denser post-optimisation replay.
    guard_band = float(torque_cfg.get("optimization_guard_band", 0.02))
    guard_band = min(max(guard_band, 0.0), 0.99)
    tau_lim_inner = np.asarray(tau_lim, dtype=float) * (1.0 - guard_band)
    cons = []

    _cx = [None]
    _ct = [None]

    def _get_traj(x):
        if _cx[0] is None or not np.array_equal(x, _cx[0]):
            _cx[0] = x.copy()
            _ct[0] = fourier_trajectory(x, freqs, t, q0, basis, opt_phase)
        return _ct[0]

    _tdx = [None]
    _tdd = [None]

    def _get_design(x):
        if _tdx[0] is None or not np.array_equal(x, _tdx[0]):
            q, dq, ddq = _get_traj(x)
            _tdx[0] = x.copy()
            _tdd[0] = compute_torque_design_data(
                q, dq, ddq, get_regressor_fn, nominal_params, tau_lim_inner,
                torque_method, torque_cfg,
            )
        return _tdd[0]

    def _make_torque_upper(idx):
        def con(x):
            d = _get_design(x)
            return np.min(d["limit_upper"][idx] - d["design_upper"][idx])
        return {"type": "ineq", "fun": con}

    def _make_torque_lower(idx):
        def con(x):
            d = _get_design(x)
            return np.min(d["design_lower"][idx] - d["limit_lower"][idx])
        return {"type": "ineq", "fun": con}

    for i in range(nDoF):
        cons.append(_make_torque_upper(i))
        cons.append(_make_torque_lower(i))

    if torque_method == "actuator_envelope" and torque_cfg.get("rms_limit_ratio") is not None:
        def _make_rms(idx):
            def con(x):
                d = _get_design(x)
                ratio = np.asarray(torque_cfg.get("rms_limit_ratio"), dtype=float)
                base = np.maximum(abs(tau_lim_inner[idx, 0]), abs(tau_lim_inner[idx, 1]))
                limit = float(ratio) * base if ratio.ndim == 0 else float(ratio[idx]) * base
                return limit - np.sqrt(np.mean(d["tau_nominal"][idx] ** 2))
            return {"type": "ineq", "fun": con}

        for i in range(nDoF):
            cons.append(_make_rms(i))

    return cons
