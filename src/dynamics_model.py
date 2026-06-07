"""Runtime dynamics-model helpers used by the pipeline and tests.

Given a rigid-body regressor function ``Y_rigid(q, dq, ddq)``, the rigid
parameter vector ``pi_rigid``, and optionally a friction parameter vector
``pi_friction`` with ``friction_model``, these helpers return

    tau = M(q) @ ddq + c(q, dq) + g(q) + tau_f(dq)

via runtime regressor evaluations. The rigid-body terms are extracted
purely from ``Y_rigid`` / ``pi_rigid`` -- friction is NEVER mixed into
M/c/g.

These helpers are NOT shipped to downstream users. The downstream
artifact is the closed-form ``dynamics_model.py`` emitted by
``src.regressor_export.export_dynamics_model_closed_form``; these
helpers serve as the reference implementation against which that
emitted module is validated.

References
----------
- Khalil & Dombre (2002), Ch. 9
- Gaz, Cognetti et al., IEEE RA-L 2019
"""
from __future__ import annotations

import numpy as np

from .friction import build_friction_regressor, friction_param_count


def gravity_vector(rigid_fn, pi_rigid, q):
    """g(q) = Y_rigid(q, 0, 0) @ pi_rigid."""
    q = np.asarray(q, dtype=float).reshape(-1)
    n = q.size
    return rigid_fn(q, np.zeros(n), np.zeros(n)) @ np.asarray(pi_rigid)


def mass_matrix(rigid_fn, pi_rigid, q, g_q=None):
    """M(q) column-by-column: M[:, j] = Y(q, 0, e_j) @ pi - g(q)."""
    q = np.asarray(q, dtype=float).reshape(-1)
    n = q.size
    if g_q is None:
        g_q = gravity_vector(rigid_fn, pi_rigid, q)
    M = np.zeros((n, n))
    pi_rigid = np.asarray(pi_rigid, dtype=float)
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        M[:, j] = rigid_fn(q, np.zeros(n), ej) @ pi_rigid - g_q
    return M


def coriolis_vector(rigid_fn, pi_rigid, q, dq, g_q=None):
    """c(q, dq) = Y_rigid(q, dq, 0) @ pi_rigid - g(q)."""
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)
    n = q.size
    if g_q is None:
        g_q = gravity_vector(rigid_fn, pi_rigid, q)
    return rigid_fn(q, dq, np.zeros(n)) @ np.asarray(pi_rigid) - g_q


def coriolis_matrix_christoffel(rigid_fn, pi_rigid, q, dq, *, fd_step=1e-6):
    """Christoffel-style Coriolis matrix C(q, dq) such that c = C @ dq."""
    q = np.asarray(q, dtype=float).reshape(-1)
    dq = np.asarray(dq, dtype=float).reshape(-1)
    n = q.size
    dM = np.zeros((n, n, n))
    for k in range(n):
        qp = q.copy()
        qp[k] += fd_step
        qm = q.copy()
        qm[k] -= fd_step
        dM[k] = (mass_matrix(rigid_fn, pi_rigid, qp)
                 - mass_matrix(rigid_fn, pi_rigid, qm)) / (2.0 * fd_step)
    # Gamma[i,j,k] = 0.5 * (dM[k][i,j] + dM[j][i,k] - dM[i][j,k])
    # dM has axes (k, i, j) → permute to assemble Christoffels.
    Gamma = 0.5 * (dM.transpose(1, 2, 0)
                   + dM.transpose(2, 1, 0)
                   - dM.transpose(0, 2, 1))
    return Gamma @ dq


def friction_torque(dq, pi_friction, friction_model):
    """tau_f(dq) = Y_friction(dq) @ pi_friction."""
    dq = np.asarray(dq, dtype=float).reshape(-1)
    if (friction_model == "none"
            or pi_friction is None
            or np.asarray(pi_friction).size == 0):
        return np.zeros(dq.size)
    Yf = build_friction_regressor(dq, friction_model)
    return Yf @ np.asarray(pi_friction, dtype=float).reshape(-1)


def split_pi_aug(pi_aug, n_dof, friction_model):
    """Split an augmented parameter vector into (pi_rigid, pi_friction)."""
    pi_aug = np.asarray(pi_aug, dtype=float).reshape(-1)
    n_rigid = 10 * n_dof
    n_fric = friction_param_count(n_dof, friction_model)
    if pi_aug.size != n_rigid + n_fric:
        raise ValueError(
            f"pi_aug has {pi_aug.size} entries but expected "
            f"{n_rigid} rigid + {n_fric} friction = {n_rigid + n_fric}."
        )
    return pi_aug[:n_rigid], pi_aug[n_rigid:]


def compute_full_dynamics(rigid_fn, pi_rigid, q, dq, *,
                          pi_friction=None, friction_model="none"):
    """Return dict with 'g', 'M', 'c', 'tau_f' at (q, dq). Shares g eval."""
    g_q = gravity_vector(rigid_fn, pi_rigid, q)
    M = mass_matrix(rigid_fn, pi_rigid, q, g_q=g_q)
    c = coriolis_vector(rigid_fn, pi_rigid, q, dq, g_q=g_q)
    tau_f = friction_torque(dq, pi_friction, friction_model)
    return {"g": g_q, "M": M, "c": c, "tau_f": tau_f}


def verify_dynamics_consistency(rigid_fn, pi_rigid, q, dq, ddq, *,
                                pi_friction=None, friction_model="none"):
    """Check tau_lin == M*ddq + c + g + tau_f to machine precision."""
    dyn = compute_full_dynamics(
        rigid_fn, pi_rigid, q, dq,
        pi_friction=pi_friction, friction_model=friction_model,
    )
    ddq = np.asarray(ddq, dtype=float).reshape(-1)
    tau_recon = dyn["M"] @ ddq + dyn["c"] + dyn["g"] + dyn["tau_f"]
    tau_rigid_lin = rigid_fn(q, dq, ddq) @ np.asarray(pi_rigid)
    tau_fric_lin = friction_torque(dq, pi_friction, friction_model)
    tau_lin = tau_rigid_lin + tau_fric_lin
    return tau_lin, tau_recon, float(np.max(np.abs(tau_lin - tau_recon)))
