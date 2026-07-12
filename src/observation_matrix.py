"""Observation-matrix construction: stack regressors over time and downsample."""
import logging

import numpy as np

from .filtering import apply_filter
from .friction import build_friction_regressor

logger = logging.getLogger("sysid_pipeline")


def build_observation_matrix(q_data: np.ndarray,
                             dq_data: np.ndarray,
                             ddq_data: np.ndarray,
                             tau_data: np.ndarray,
                             regressor_fn,
                             cfg: dict,
                             original_fs: float,
                             return_metadata: bool = False) -> tuple:
    """Build the stacked observation matrix W and torque vector.

    Parameters
    ----------
    q_data, dq_data, ddq_data : (N, nDoF)
    tau_data : (N, nDoF)
    regressor_fn : callable(q, dq, ddq) -> Y (nDoF x p_rigid)
    cfg : full pipeline config
    original_fs : sampling frequency of input data [Hz]
    return_metadata : bool
        When True, also return the filtered/downsampled samples used to build
        W and tau_vec. This supports observation-matrix cache validation.

    Returns
    -------
    W : (M*nDoF, p) stacked observation matrix
    tau_vec : (M*nDoF,) stacked torque vector
    samples : dict, optional
    """
    friction_model = cfg["friction"]["model"]
    samples = prepare_observation_samples(
        q_data, dq_data, ddq_data, tau_data, cfg, original_fs
    )
    q_used = samples["q"]
    dq_used = samples["dq"]
    ddq_used = samples["ddq"]
    tau_used = samples["tau"]
    M, nDoF = q_used.shape

    logger.info(
        "Building observation matrix: %d time samples, nDoF=%d", M, nDoF
    )

    Y0 = np.asarray(regressor_fn(q_used[0], dq_used[0], ddq_used[0]), dtype=float)
    if friction_model != "none":
        Yf0 = build_friction_regressor(dq_used[0], friction_model)
        p = Y0.shape[1] + Yf0.shape[1]
    else:
        p = Y0.shape[1]

    W = np.zeros((M * nDoF, p))
    tau_vec = np.zeros(M * nDoF)

    for k in range(M):
        Y = np.asarray(regressor_fn(q_used[k], dq_used[k], ddq_used[k]), dtype=float)
        if friction_model != "none":
            Yf = build_friction_regressor(dq_used[k], friction_model)
            Y = np.hstack((Y, Yf))
        W[k * nDoF:(k + 1) * nDoF, :] = Y
        tau_vec[k * nDoF:(k + 1) * nDoF] = tau_used[k]

    samples["tau_vec"] = tau_vec

    logger.info(
        "Observation matrix W: shape %s, overdetermination ratio %.1f",
        W.shape,
        M * nDoF / p,
    )

    n_equations = M * nDoF
    if n_equations < p:
        raise ValueError(
            f"Insufficient data: {n_equations} equations < {p} unknowns. "
            f"Need at least {p} stacked rows (got {M} time samples x "
            f"{nDoF} DoF = {n_equations}). Increase trajectory duration "
            f"or reduce downsampling."
        )
    if n_equations < 2 * p:
        logger.warning(
            "Low overdetermination: only %.1fx (%d equations for %d unknowns). "
            "Identification may be poorly conditioned.",
            n_equations / p,
            n_equations,
            p,
        )

    if return_metadata:
        return W, tau_vec, samples
    return W, tau_vec


def prepare_observation_samples(q_data: np.ndarray,
                                dq_data: np.ndarray,
                                ddq_data: np.ndarray,
                                tau_data: np.ndarray,
                                cfg: dict,
                                original_fs: float) -> dict:
    """Return the exact filtered/downsampled samples used to build W."""
    N, nDoF = q_data.shape

    q_f = apply_filter(q_data, original_fs, cfg["filtering"])
    dq_f = apply_filter(dq_data, original_fs, cfg["filtering"])
    ddq_f = apply_filter(ddq_data, original_fs, cfg["filtering"])
    tau_f = apply_filter(tau_data, original_fs, cfg["filtering"])

    ds_freq = cfg["downsampling"]["frequency_hz"]
    if ds_freq > 0 and ds_freq < original_fs:
        step = max(1, int(round(original_fs / ds_freq)))
        # Integer decimation cannot hit an arbitrary requested rate; the
        # ACTUAL retained rate governs the new Nyquist band.
        actual_fs = original_fs / step
        if abs(actual_fs - ds_freq) > 1e-9 * max(ds_freq, 1.0):
            logger.info(
                "Downsampling: %g Hz -> requested %g Hz, actual %g Hz "
                "(step=%d)", original_fs, ds_freq, actual_fs, step,
            )
        else:
            logger.info(
                "Downsampling: %d Hz -> %d Hz (step=%d)",
                int(original_fs), int(ds_freq), step,
            )
        # Decimation without a matching low-pass is aliasing: everything
        # above the new Nyquist folds back into the identification band.
        cfg_filter = cfg.get("filtering", {}) or {}
        anti_alias_cutoff = 0.45 * actual_fs
        if step > 1 and not cfg_filter.get("enabled", False):
            logger.warning(
                "Downsampling %g Hz -> %g Hz WITHOUT low-pass filtering: "
                "signal/noise content above %g Hz will alias into the "
                "identification band. Enable filtering with "
                "cutoff_frequency_hz <= %g.",
                original_fs, actual_fs, 0.5 * actual_fs, anti_alias_cutoff,
            )
        elif step > 1 and float(
            cfg_filter.get("cutoff_frequency_hz", 0.0) or 0.0
        ) > anti_alias_cutoff:
            logger.warning(
                "Filter cutoff %g Hz exceeds the anti-aliasing limit %g Hz "
                "for downsampling to %g Hz; residual content between the "
                "cutoff and the new Nyquist (%g Hz) will alias.",
                float(cfg_filter.get("cutoff_frequency_hz")),
                anti_alias_cutoff, actual_fs, 0.5 * actual_fs,
            )
    else:
        step = 1

    indices = np.arange(0, N, step, dtype=np.int64)
    tau_used = tau_f[indices]
    return {
        "q": q_f[indices],
        "dq": dq_f[indices],
        "ddq": ddq_f[indices],
        "tau": tau_used,
        "tau_vec": tau_used.reshape(-1),
        "sample_indices": indices,
        "step": int(step),
        "nDoF": int(nDoF),
        "original_fs": float(original_fs),
    }
