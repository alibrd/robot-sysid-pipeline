"""Observation-matrix construction: stack regressors over time and downsample."""
import logging
import numpy as np

from .friction import build_friction_regressor
from .filtering import apply_filter

logger = logging.getLogger("sysid_pipeline")


def build_observation_matrix(q_data: np.ndarray,
                             dq_data: np.ndarray,
                             ddq_data: np.ndarray,
                             tau_data: np.ndarray,
                             regressor_fn,
                             cfg: dict,
                             original_fs: float) -> tuple:
    """Build the stacked observation matrix W and torque vector.

    Parameters
    ----------
    q_data, dq_data, ddq_data : (N, nDoF)
    tau_data : (N, nDoF)
    regressor_fn : callable(q, dq, ddq) -> Y (nDoF × p)
    cfg : full pipeline config
    original_fs : sampling frequency of input data [Hz]

    Returns
    -------
    W : (M*nDoF, p) stacked observation matrix
    tau_vec : (M*nDoF,) stacked torque vector
    """
    N, nDoF = q_data.shape
    friction_model = cfg["friction"]["model"]

    # ── Filtering ──────────────────────────────────────────────────────
    q_f = apply_filter(q_data, original_fs, cfg["filtering"])
    dq_f = apply_filter(dq_data, original_fs, cfg["filtering"])
    ddq_f = apply_filter(ddq_data, original_fs, cfg["filtering"])
    tau_f = apply_filter(tau_data, original_fs, cfg["filtering"])

    # ── Downsampling ───────────────────────────────────────────────────
    ds_freq = cfg["downsampling"]["frequency_hz"]
    if ds_freq > 0 and ds_freq < original_fs:
        step = max(1, int(round(original_fs / ds_freq)))
        logger.info("Downsampling: %d Hz → %d Hz (step=%d)", int(original_fs), int(ds_freq), step)
    else:
        step = 1

    indices = list(range(0, N, step))
    M = len(indices)

    logger.info("Building observation matrix: %d time samples, nDoF=%d", M, nDoF)

    # Determine column count from first sample
    Y0 = regressor_fn(q_f[indices[0]], dq_f[indices[0]], ddq_f[indices[0]])
    if friction_model != "none":
        Yf0 = build_friction_regressor(dq_f[indices[0]], friction_model)
        p = Y0.shape[1] + Yf0.shape[1]
    else:
        p = Y0.shape[1]

    W = np.zeros((M * nDoF, p))
    tau_vec = np.zeros(M * nDoF)

    for k, idx in enumerate(indices):
        Y = regressor_fn(q_f[idx], dq_f[idx], ddq_f[idx])
        if friction_model != "none":
            Yf = build_friction_regressor(dq_f[idx], friction_model)
            Y = np.hstack((Y, Yf))
        W[k * nDoF:(k + 1) * nDoF, :] = Y
        tau_vec[k * nDoF:(k + 1) * nDoF] = tau_f[idx]

    logger.info("Observation matrix W: shape %s, overdetermination ratio %.1f",
                W.shape, M * nDoF / p)

    # ── Sample sufficiency check ───────────────────────────────────────
    n_equations = M * nDoF
    if n_equations < p:
        raise ValueError(
            f"Insufficient data: {n_equations} equations < {p} unknowns. "
            f"Need at least {p} stacked rows (got {M} time samples × "
            f"{nDoF} DoF = {n_equations}). Increase trajectory duration "
            f"or reduce downsampling."
        )
    if n_equations < 2 * p:
        logger.warning(
            "Low overdetermination: only %.1fx (%d equations for %d unknowns). "
            "Identification may be poorly conditioned.",
            n_equations / p, n_equations, p
        )

    return W, tau_vec
