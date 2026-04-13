"""Friction regressor construction: viscous, Coulomb, or both.

Extends the rigid-body regressor Y (n×10n) with friction columns to produce
the augmented regressor [Y | Y_friction] and the augmented parameter vector.
"""
import numpy as np


def build_friction_regressor(dq: np.ndarray, friction_model: str,
                             sigmoid_a: float = 10.0,
                             sigmoid_b: float = 1000.0) -> np.ndarray:
    """Build friction regressor columns for a single time step.

    Parameters
    ----------
    dq : (n,) joint velocities
    friction_model : "none" | "viscous" | "coulomb" | "viscous_coulomb"
    sigmoid_a, sigmoid_b : Coulomb sigmoid smoothing parameters

    Returns
    -------
    Y_fric : (n, n_fric_params) friction regressor block.
             n_fric_params = 0 / n / 2n / 3n depending on model.
    """
    n = dq.size

    if friction_model == "none":
        return np.zeros((n, 0))

    blocks = []

    if friction_model in ("viscous", "viscous_coulomb"):
        # Y_v = diag(dq)  →  n × n
        Y_v = np.diag(dq)
        blocks.append(Y_v)

    if friction_model in ("coulomb", "viscous_coulomb"):
        # Positive-direction Coulomb: sigmoid(a - b*dq_i)
        Y_cp = np.diag(1.0 / (1.0 + np.exp(sigmoid_a - sigmoid_b * dq)))
        # Negative-direction Coulomb: -sigmoid(a + b*dq_i)
        Y_cn = np.diag(-1.0 / (1.0 + np.exp(sigmoid_a + sigmoid_b * dq)))
        blocks.append(Y_cp)
        blocks.append(Y_cn)

    return np.hstack(blocks)


def friction_param_count(n: int, friction_model: str) -> int:
    """Return number of friction parameters for *n* joints."""
    if friction_model == "none":
        return 0
    if friction_model == "viscous":
        return n
    if friction_model == "coulomb":
        return 2 * n
    if friction_model == "viscous_coulomb":
        return 3 * n
    return 0


def friction_param_names(n: int, friction_model: str):
    """Return list of human-readable parameter names."""
    names = []
    if friction_model in ("viscous", "viscous_coulomb"):
        names += [f"dv_{i+1}" for i in range(n)]
    if friction_model in ("coulomb", "viscous_coulomb"):
        names += [f"dcp_{i+1}" for i in range(n)]
        names += [f"dcn_{i+1}" for i in range(n)]
    return names
