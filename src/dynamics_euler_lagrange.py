"""Euler-Lagrange symbolic dynamics: regressor matrix construction.

Builds the regressor symbolically via the Lagrangian approach (energy-based)
using SymPy. The symbolic regressor is cached as a SymPy expression (NOT a
closure) to a pickle file and re-lambdified on load.

References:
  Khalil & Dombre (2002), ch. 9 -- energy-based regressor derivation.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import sympy

from .math_utils import skew_sym, axis_rotation_sym, GRAVITY_SI
from .kinematics import RobotKinematics

logger = logging.getLogger("sysid_pipeline")


def euler_lagrange_regressor_builder(kin: RobotKinematics, cache_dir: str):
    """Build (or load cached) symbolic EL regressor.

    Returns
    -------
    regressor_fn : callable(q, dq, ddq) -> Y (nDoF x p_kept)
    kept_cols : list of column indices retained from the full 10*nDoF vector
    """
    n = kin.nDoF
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    pkl = cache_path / "el_regressor_cache.pkl"

    q_syms = [sympy.Symbol(f"q{i+1}") for i in range(n)]
    dq_syms = [sympy.Symbol(f"dq{i+1}") for i in range(n)]
    ddq_syms = [sympy.Symbol(f"ddq{i+1}") for i in range(n)]
    all_syms = q_syms + dq_syms + ddq_syms

    if pkl.exists():
        logger.info("Loading cached EL regressor from %s", pkl)
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        Y_sym = data["Y_sym"]
        kept_cols = data["kept_cols"]
    else:
        logger.info("Building symbolic EL regressor for %d DoF ...", n)
        Y_sym, kept_cols = _build_symbolic_regressor(kin, q_syms, dq_syms, ddq_syms)
        with open(pkl, "wb") as f:
            pickle.dump({"Y_sym": Y_sym, "kept_cols": kept_cols}, f)
        logger.info("Cached EL regressor to %s", pkl)

    logger.info("EL regressor: %d x %d (kept %d of %d columns)",
                Y_sym.shape[0], Y_sym.shape[1], len(kept_cols), 10 * n)

    # Lambdify (fresh each time -- closures are NEVER pickled)
    Y_fn = sympy.lambdify(all_syms, Y_sym, modules="numpy")

    def regressor_fn(q_val, dq_val, ddq_val):
        args = list(q_val) + list(dq_val) + list(ddq_val)
        raw = Y_fn(*args)
        out = np.atleast_2d(np.array(raw, dtype=float))
        if out.shape[0] != n:
            out = out.reshape(n, -1)
        return out

    return regressor_fn, kept_cols


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_sym_transform(kin, link_idx, q_sym):
    """Rebuild symbolic T_{i-1}^{i} for *link_idx* from numeric data in *kin*.

    Uses di_raw (origin_xyz of this joint in parent frame) as translation.
    """
    di = sympy.Matrix(kin.di_raw[link_idx]).reshape(3, 1)

    # Recover R0 by evaluating the lambdified transform at q = 0
    Ti_at_zero = np.array(kin.link_kin[link_idx].Ti_1_i(0.0)).reshape(4, 4)
    R0 = sympy.Matrix(Ti_at_zero[:3, :3])

    # Recover axis from torque_row_sign
    row, sign = kin.torque_row_sign[link_idx]
    axis_vec = [0.0, 0.0, 0.0]
    axis_vec[row - 3] = sign
    Rq = axis_rotation_sym(axis_vec, q_sym)

    R = R0 * Rq
    top = R.row_join(di)
    return top.col_join(sympy.Matrix([[0, 0, 0, 1]]))


def _build_symbolic_regressor(kin, q_syms, dq_syms, ddq_syms):
    """Derive the n x 10n regressor from the Lagrangian."""
    n = kin.nDoF
    g_w = sympy.Matrix([0, 0, -GRAVITY_SI])  # world-frame gravity (3x1)

    # Absolute transforms T_world_i
    Tw = [sympy.Matrix(kin.Tw_0)]
    for i in range(n):
        Ti_sym = _build_sym_transform(kin, i, q_syms[i])
        Tw.append(Tw[-1] * Ti_sym)

    dq_vec = sympy.Matrix(dq_syms)

    # Build 1 x 10n Lagrangian coefficient row (Y_L)
    Y_L = sympy.zeros(1, 10 * n)

    for i in range(n):
        Rw_i = Tw[i + 1][:3, :3]
        pw_i = Tw[i + 1][:3, 3]

        # World-frame Jacobians
        Jv_w = pw_i.jacobian(sympy.Matrix(q_syms))          # 3 x n
        dRw = sympy.zeros(3, 3)
        for j in range(n):
            dRw += Rw_i.diff(q_syms[j]) * dq_syms[j]
        Sw = dRw * Rw_i.T
        ww = sympy.Matrix([Sw[2, 1], Sw[0, 2], Sw[1, 0]])  # world omega

        Jw_w = sympy.zeros(3, n)
        for j in range(n):
            dR_j = Rw_i.diff(q_syms[j])
            S_j = dR_j * Rw_i.T
            Jw_w[0, j] = S_j[2, 1]
            Jw_w[1, j] = S_j[0, 2]
            Jw_w[2, j] = S_j[1, 0]

        # Body-frame velocities
        vw = Jv_w * dq_vec       # 3x1
        vi = Rw_i.T * vw         # body-frame linear velocity of origin
        wi = Rw_i.T * ww         # body-frame angular velocity

        # Body-frame gravity
        gb = Rw_i.T * g_w        # 3x1

        # ---- Lagrangian coefficients for link i (body-frame formulation) ----
        # L_i = T_i - V_i  linear in pi_i = [m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        #
        # V = -m * g_w^T * p_CoM_world, so L includes + g_w^T * pw for mass
        # and + g_body^T * h for first moment (L = T - V = T + g_w^T * p_CoM)
        coeff_m = (sympy.Rational(1, 2) * (vi.T * vi)[0, 0]
                   + (g_w.T * pw_i)[0, 0])

        # first moment h = [mx, my, mz]:
        #   KE part:  cross(vi, wi) . h
        #   PE part: +g_body^T * h  (since L = T - V and V = -g_w^T*Rw*h)
        cross_vw = sympy.Matrix([
            vi[1] * wi[2] - vi[2] * wi[1],
            vi[2] * wi[0] - vi[0] * wi[2],
            vi[0] * wi[1] - vi[1] * wi[0],
        ])
        coeff_mx = cross_vw[0] + gb[0]
        coeff_my = cross_vw[1] + gb[1]
        coeff_mz = cross_vw[2] + gb[2]

        # inertia [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]:
        #   0.5 * wi^T I wi expanded
        wx, wy, wz = wi[0], wi[1], wi[2]
        coeff_Ixx = sympy.Rational(1, 2) * wx**2
        coeff_Ixy = wx * wy
        coeff_Ixz = wx * wz
        coeff_Iyy = sympy.Rational(1, 2) * wy**2
        coeff_Iyz = wy * wz
        coeff_Izz = sympy.Rational(1, 2) * wz**2

        base = 10 * i
        Y_L[0, base + 0] = coeff_m
        Y_L[0, base + 1] = coeff_mx
        Y_L[0, base + 2] = coeff_my
        Y_L[0, base + 3] = coeff_mz
        Y_L[0, base + 4] = coeff_Ixx
        Y_L[0, base + 5] = coeff_Ixy
        Y_L[0, base + 6] = coeff_Ixz
        Y_L[0, base + 7] = coeff_Iyy
        Y_L[0, base + 8] = coeff_Iyz
        Y_L[0, base + 9] = coeff_Izz

    # Apply Euler-Lagrange equation to get regressor
    Y_reg = _differentiate_lagrangian(Y_L, q_syms, dq_syms, ddq_syms)
    Y_reg = sympy.nsimplify(Y_reg, rational=False)

    # Remove zero columns
    Y_reg, kept_cols = _remove_zero_columns(Y_reg)

    return Y_reg, kept_cols


def _differentiate_lagrangian(Y_L, q, dq, ddq):
    """Derive regressor Y (n x p) from Lagrangian row Y_L (1 x p).

    tau_j = d/dt(dL/d(dq_j)) - dL/dq_j  =>  Y[j,:] = sum expansions.
    """
    n = len(q)
    p = Y_L.shape[1]

    # dL/dq  (n x p)
    Y_q = sympy.zeros(n, p)
    for i in range(n):
        for j in range(p):
            Y_q[i, j] = sympy.diff(Y_L[0, j], q[i])

    # dL/d(dq)  (n x p)
    Y_dq = sympy.zeros(n, p)
    for i in range(n):
        for j in range(p):
            Y_dq[i, j] = sympy.diff(Y_L[0, j], dq[i])

    # d/dt(dL/d(dq))
    # = sum_k dq_k * d(Y_dq)/dq_k  +  sum_k ddq_k * d(Y_dq)/d(dq_k)
    Ydq_dot = sympy.zeros(n, p)
    for k in range(n):
        Ydq_dot += dq[k] * sympy.Matrix(
            [[sympy.diff(Y_dq[i, j], q[k]) for j in range(p)] for i in range(n)]
        )
        Ydq_dot += ddq[k] * sympy.Matrix(
            [[sympy.diff(Y_dq[i, j], dq[k]) for j in range(p)] for i in range(n)]
        )

    return Ydq_dot - Y_q


def _remove_zero_columns(Y):
    """Drop columns that are identically zero, returning (Y_reduced, kept_indices)."""
    kept = []
    for j in range(Y.shape[1]):
        if not Y[:, j].equals(sympy.zeros(Y.shape[0], 1)):
            kept.append(j)
    if not kept:
        return Y, list(range(Y.shape[1]))
    # Build reduced matrix by horizontal concatenation of kept columns
    result = Y[:, kept[0]]
    for k in kept[1:]:
        result = result.row_join(Y[:, k])
    return result, kept
