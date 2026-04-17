"""Shared helpers for torque-limited excitation design and replay validation."""
import math
from statistics import NormalDist

import numpy as np

from .friction import build_friction_regressor, friction_param_count


VALID_TORQUE_METHODS = {
    "none",
    "nominal_hard",
    "soft_penalty",
    "robust_box",
    "chance",
    "actuator_envelope",
    "sequential_redesign",
}
VALID_ENVELOPE_TYPES = {"constant", "speed_linear"}
HARD_TORQUE_METHODS = {"nominal_hard", "robust_box", "chance", "actuator_envelope"}


def build_nominal_parameter_vector(pi_rigid: np.ndarray,
                                   nDoF: int,
                                   friction_model: str,
                                   friction_nominal: np.ndarray | None = None) -> np.ndarray:
    """Append nominal friction parameters to a rigid-body parameter vector."""
    pi_rigid = np.asarray(pi_rigid, dtype=float).reshape(-1)
    n_fric = friction_param_count(nDoF, friction_model)
    if n_fric == 0:
        return pi_rigid.copy()
    if friction_nominal is None:
        friction_nominal = np.zeros(n_fric, dtype=float)
    friction_nominal = np.asarray(friction_nominal, dtype=float).reshape(-1)
    if friction_nominal.size != n_fric:
        raise ValueError(
            f"Expected {n_fric} nominal friction parameters, got {friction_nominal.size}."
        )
    return np.concatenate([pi_rigid, friction_nominal])


def make_augmented_regressor(base_regressor_fn, friction_model: str):
    """Return a regressor callable that appends friction columns when requested."""
    def get_regressor(q_val, dq_val, ddq_val):
        Y = base_regressor_fn(q_val, dq_val, ddq_val)
        if friction_model != "none":
            Yf = build_friction_regressor(np.asarray(dq_val, dtype=float), friction_model)
            Y = np.hstack((Y, Yf))
        return Y
    return get_regressor


def evaluate_torque_series(q: np.ndarray,
                           dq: np.ndarray,
                           ddq: np.ndarray,
                           get_regressor_fn,
                           params: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Evaluate torque and per-sample regressors on a trajectory.

    Parameters use the trajectory convention (nDoF, N).
    """
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    ddq = np.asarray(ddq, dtype=float)
    params = np.asarray(params, dtype=float).reshape(-1)

    if q.shape != dq.shape or q.shape != ddq.shape:
        raise ValueError("q, dq, and ddq must share the same shape (nDoF, N).")

    n_dof, n_samples = q.shape
    tau = np.zeros((n_dof, n_samples), dtype=float)
    regressors = []
    for idx in range(n_samples):
        Y = np.asarray(get_regressor_fn(q[:, idx], dq[:, idx], ddq[:, idx]), dtype=float)
        tau[:, idx] = Y @ params
        regressors.append(Y)
    return tau, regressors


def compute_actual_torque_limits(dq: np.ndarray,
                                 tau_lim: np.ndarray,
                                 method: str,
                                 torque_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper torque limits at each sample."""
    dq = np.asarray(dq, dtype=float)
    tau_lim = np.asarray(tau_lim, dtype=float)
    lower = np.repeat(tau_lim[:, 0][:, None], dq.shape[1], axis=1)
    upper = np.repeat(tau_lim[:, 1][:, None], dq.shape[1], axis=1)

    if method != "actuator_envelope":
        return lower, upper

    env_type = torque_cfg.get("envelope_type", "constant")
    if env_type == "constant":
        return lower, upper
    if env_type != "speed_linear":
        raise ValueError(f"Unsupported actuator envelope type: {env_type}")

    velocity_reference = float(torque_cfg.get("velocity_reference", 1.0))
    slope = float(torque_cfg.get("speed_linear_slope", 0.0))
    min_scale = float(torque_cfg.get("min_scale", 0.5))
    max_scale = float(torque_cfg.get("max_scale", 1.5))
    scale = 1.0 - slope * np.abs(dq) / max(velocity_reference, 1e-12)
    scale = np.clip(scale, min_scale, max_scale)
    return lower * scale, upper * scale


def compute_torque_design_data(q: np.ndarray,
                               dq: np.ndarray,
                               ddq: np.ndarray,
                               get_regressor_fn,
                               nominal_params: np.ndarray,
                               tau_lim: np.ndarray,
                               method: str,
                               torque_cfg: dict) -> dict:
    """Compute actual and design-time torque bounds for a trajectory."""
    tau_nominal, regressors = evaluate_torque_series(q, dq, ddq, get_regressor_fn, nominal_params)
    limit_lower, limit_upper = compute_actual_torque_limits(dq, tau_lim, method, torque_cfg)
    design_lower = tau_nominal.copy()
    design_upper = tau_nominal.copy()
    quantile = None

    if method == "robust_box":
        delta = _parameter_uncertainty_radius(
            nominal_params,
            float(torque_cfg.get("relative_uncertainty", 0.1)),
            float(torque_cfg.get("absolute_uncertainty_floor", 1e-3)),
        )
        margin = np.zeros_like(tau_nominal)
        for idx, Y in enumerate(regressors):
            margin[:, idx] = np.abs(Y) @ delta
        design_lower = tau_nominal - margin
        design_upper = tau_nominal + margin

    elif method == "chance":
        sigma = _parameter_uncertainty_radius(
            nominal_params,
            float(torque_cfg.get("relative_stddev", 0.05)),
            float(torque_cfg.get("absolute_stddev_floor", 1e-3)),
        )
        confidence = float(torque_cfg.get("chance_confidence", 0.99))
        quantile = NormalDist().inv_cdf(confidence)
        margin = np.zeros_like(tau_nominal)
        sigma_sq = sigma**2
        for idx, Y in enumerate(regressors):
            margin[:, idx] = quantile * np.sqrt((Y**2) @ sigma_sq)
        design_lower = tau_nominal - margin
        design_upper = tau_nominal + margin

    upper_margin = limit_upper - design_upper
    lower_margin = design_lower - limit_lower
    design_pass = bool(np.all(upper_margin >= -1e-10) and np.all(lower_margin >= -1e-10))

    actual_upper_margin = limit_upper - tau_nominal
    actual_lower_margin = tau_nominal - limit_lower
    actual_pass = bool(np.all(actual_upper_margin >= -1e-10) and np.all(actual_lower_margin >= -1e-10))

    return {
        "tau_nominal": tau_nominal,
        "limit_lower": limit_lower,
        "limit_upper": limit_upper,
        "design_lower": design_lower,
        "design_upper": design_upper,
        "design_upper_margin": upper_margin,
        "design_lower_margin": lower_margin,
        "actual_upper_margin": actual_upper_margin,
        "actual_lower_margin": actual_lower_margin,
        "design_pass": design_pass,
        "actual_pass": actual_pass,
        "quantile": quantile,
    }


def compute_soft_penalty(tau: np.ndarray,
                         lower: np.ndarray,
                         upper: np.ndarray,
                         weight: float,
                         smoothing: float) -> float:
    """Return a smooth squared violation penalty for torque bounds."""
    smoothing = max(float(smoothing), 1e-12)
    upper_violation = _softplus((tau - upper) / smoothing) * smoothing
    lower_violation = _softplus((lower - tau) / smoothing) * smoothing
    penalty = np.sum(upper_violation**2 + lower_violation**2)
    return float(weight) * float(penalty)


def compute_rms_limits(tau_lim: np.ndarray, torque_cfg: dict) -> np.ndarray | None:
    """Return per-joint RMS torque limits if configured."""
    ratio = torque_cfg.get("rms_limit_ratio")
    if ratio is None:
        return None
    tau_lim = np.asarray(tau_lim, dtype=float)
    base = np.maximum(np.abs(tau_lim[:, 0]), np.abs(tau_lim[:, 1]))
    ratio_arr = np.asarray(ratio, dtype=float)
    if ratio_arr.ndim == 0:
        ratio_arr = np.full(base.shape, float(ratio_arr))
    if ratio_arr.shape != base.shape:
        raise ValueError(
            f"rms_limit_ratio must be scalar or shape {base.shape}, got {ratio_arr.shape}."
        )
    return ratio_arr * base


def summarize_torque_replay(tau: np.ndarray,
                            lower: np.ndarray,
                            upper: np.ndarray,
                            rms_limits: np.ndarray | None = None) -> dict:
    """Summarize torque-limit compliance for a replay."""
    tau = np.asarray(tau, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    upper_margin = upper - tau
    lower_margin = tau - lower
    ratio = normalized_torque_ratio(tau, lower, upper)
    worst_index = np.unravel_index(np.argmax(ratio), ratio.shape)
    rms_torque = np.sqrt(np.mean(tau**2, axis=1))
    rms_pass = True
    if rms_limits is not None:
        rms_limits = np.asarray(rms_limits, dtype=float).reshape(-1)
        rms_pass = bool(np.all(rms_torque <= rms_limits + 1e-10))

    return {
        "pass": bool(np.all(upper_margin >= -1e-10) and np.all(lower_margin >= -1e-10) and rms_pass),
        "upper_margin": upper_margin,
        "lower_margin": lower_margin,
        "ratio": ratio,
        "max_ratio": float(np.max(ratio)),
        "worst_joint": int(worst_index[0]),
        "worst_time_index": int(worst_index[1]),
        "rms_torque": rms_torque,
        "rms_limits": rms_limits,
        "rms_pass": rms_pass,
    }


def replay_torque_models(q: np.ndarray,
                         dq: np.ndarray,
                         ddq: np.ndarray,
                         get_regressor_fn,
                         tau_lim: np.ndarray,
                         method: str,
                         torque_cfg: dict,
                         *,
                         nominal_params: np.ndarray | None = None,
                         identified_params: np.ndarray | None = None,
                         corrected_params: np.ndarray | None = None) -> dict:
    """Replay one or more models against the torque limits."""
    actual_lower, actual_upper = compute_actual_torque_limits(dq, tau_lim, method, torque_cfg)
    rms_limits = compute_rms_limits(tau_lim, torque_cfg) if method == "actuator_envelope" else None
    out = {
        "limit_lower": actual_lower,
        "limit_upper": actual_upper,
    }

    for label, params in (
        ("nominal", nominal_params),
        ("identified", identified_params),
        ("corrected", corrected_params),
    ):
        if params is None:
            continue
        tau, _ = evaluate_torque_series(q, dq, ddq, get_regressor_fn, params)
        summary = summarize_torque_replay(tau, actual_lower, actual_upper, rms_limits=rms_limits)
        out[f"tau_{label}"] = tau
        out[f"{label}_summary"] = summary
    return out


def normalized_torque_ratio(tau: np.ndarray,
                            lower: np.ndarray,
                            upper: np.ndarray) -> np.ndarray:
    """Normalize torque magnitude against the active lower/upper limit."""
    tau = np.asarray(tau, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    ratio = np.zeros_like(tau)

    abs_cap = np.maximum(np.abs(lower), np.abs(upper))
    pos_den = np.where(upper > 0.0, upper, abs_cap)
    neg_den = np.where(lower < 0.0, -lower, abs_cap)

    pos_mask = tau >= 0.0
    ratio[pos_mask] = np.divide(
        tau[pos_mask],
        np.maximum(pos_den[pos_mask], 1e-12),
    )
    ratio[~pos_mask] = np.divide(
        -tau[~pos_mask],
        np.maximum(neg_den[~pos_mask], 1e-12),
    )
    return ratio


def validation_time_vector(freqs: np.ndarray,
                           base_frequency_hz: float,
                           trajectory_duration_periods: float,
                           oversample_factor: float) -> np.ndarray:
    """Return a dense replay grid for post-optimization/post-identification checks."""
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    oversample = max(float(oversample_factor), 1.0)
    tf = float(trajectory_duration_periods) / float(base_frequency_hz)
    dt = 1.0 / (2.0 * float(np.max(freqs)) * oversample)
    return np.arange(0.0, tf + dt, dt)


def _parameter_uncertainty_radius(params: np.ndarray,
                                  relative_scale: float,
                                  absolute_floor: float) -> np.ndarray:
    params = np.asarray(params, dtype=float).reshape(-1)
    return np.maximum(np.abs(params) * float(relative_scale), float(absolute_floor))


def _softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
