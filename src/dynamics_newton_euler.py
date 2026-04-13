"""Newton-Euler recursive dynamics: numeric regressor matrix construction.

Implements the forward-pass recursive Newton-Euler formulation with adjoint-
map backward pass for building the linear regressor  tau = Y(q,dq,ddq) pi.
"""
import numpy as np
from .math_utils import skew_np, GRAVITY
from .kinematics import RobotKinematics


def newton_euler_regressor(kin: RobotKinematics,
                           q: np.ndarray,
                           dq: np.ndarray,
                           ddq: np.ndarray) -> np.ndarray:
    """Return the regressor matrix Y (nDoF x 10*nDoF) for a single time sample.

    Uses the Newton-Euler recursive forward pass (angular/linear velocities and
    accelerations in body frame) and backward adjoint-map assembly.
    """
    n = kin.nDoF
    g = GRAVITY.reshape(3, 1)

    w = np.zeros((3, n))
    dw = np.zeros((3, n))
    dv_origin = np.zeros((3, n))     # accel at frame origin (for regressor)
    dv_transport = np.zeros((3, n))  # accel at next-joint point (for propagation)
    A = np.zeros((6, 6, max(n - 1, 1)))
    reg = np.zeros((6, 10, n))

    # Store transforms for adjoint computation
    Ti_all = [None] * n
    p_zero = np.zeros((3, 1))

    for i in range(n):
        lk = kin.link_kin[i]
        Ti = np.array(lk.Ti_1_i(q[i])).reshape(4, 4)
        Ti_all[i] = Ti
        Ri = Ti[:3, :3]
        aJ = np.array(lk.aJi_1_i(q[i])).reshape(3, 1)
        pR = np.array(lk.pRi_1_i(q[i])).reshape(3, 3)
        paJ = np.array(lk.paJi_1_i(q[i])).reshape(3, 1)
        p = np.array(kin.pi_i_raw[i]).reshape(3, 1)

        if i == 0:
            w[:, 0] = (Ri.T @ (dq[0] * aJ)).flatten()
            dw[:, 0] = (
                dq[i]**2 * (pR.T @ aJ + Ri.T @ paJ)
                + ddq[i] * Ri.T @ aJ
            ).flatten()
            # Acceleration at frame origin
            dv_origin[:, 0] = (
                Ri.T @ kin.Tw_0[:3, :3].T @ (-g)
            ).flatten()
            # Transport to next-joint position for propagation
            dv_transport[:, 0] = (
                dv_origin[:, 0].reshape(3, 1)
                + skew_np(dw[:, 0]) @ p
                + skew_np(w[:, 0]) @ (skew_np(w[:, 0]) @ p)
            ).flatten()
        else:
            w[:, i] = (
                Ri.T @ (w[:, i-1].reshape(3, 1) + dq[i] * aJ)
            ).flatten()
            dw[:, i] = (
                Ri.T @ dw[:, i-1].reshape(3, 1)
                + dq[i] * pR.T @ w[:, i-1].reshape(3, 1)
                + dq[i]**2 * (pR.T @ aJ + Ri.T @ paJ)
                + ddq[i] * Ri.T @ aJ
            ).flatten()
            # Acceleration at frame origin (dv_transport[i-1] is at O_i)
            dv_origin[:, i] = (
                Ri.T @ dv_transport[:, i-1].reshape(3, 1)
            ).flatten()
            # Transport to next-joint position
            dv_transport[:, i] = (
                dv_origin[:, i].reshape(3, 1)
                + skew_np(dw[:, i]) @ p
                + skew_np(w[:, i]) @ (skew_np(w[:, i]) @ p)
            ).flatten()
            # Adjoint uses the TRANSFORM from frame i to frame i-1
            A[:, :, i-1] = _adjoint(Ti)

        reg[:, :, i] = _link_regressor(p_zero, dv_origin[:, i], dw[:, i], w[:, i])

    # -- Backward assembly ---------------------------------------------------
    Y = np.zeros((n, 10 * n))
    for rw in range(n):
        row_idx, sgn = kin.torque_row_sign[rw]
        row_block = np.zeros((6, 0))
        for cl in range(n):
            if cl < rw:
                row_block = np.hstack((row_block, np.zeros((6, 10))))
            elif cl == rw:
                row_block = np.hstack((row_block, reg[:, :, cl]))
            else:
                tmp = np.eye(6)
                for j in range(rw, cl):
                    tmp = tmp @ A[:, :, j]
                row_block = np.hstack((row_block, tmp @ reg[:, :, cl]))
        Y[rw, :] = sgn * row_block[row_idx, :]

    return Y


# -- Internal helpers -------------------------------------------------------

def _adjoint(T):
    """6x6 wrench adjoint for propagating from child to parent frame.

    Uses the rotation and translation embedded in T_{parent<-child}.
    """
    R = T[:3, :3]
    d = T[:3, 3]      # position of child origin in parent coords
    S = skew_np(d)
    top = np.hstack((R, np.zeros((3, 3))))
    bot = np.hstack((S @ R, R))
    return np.vstack((top, bot))


def _link_regressor(p, dv, dw, w):
    """Build the 6×10 per-link regressor [Y_m | Y_mp | Y_I]."""
    p = np.asarray(p).flatten()
    dv = np.asarray(dv).reshape(3, 1)
    dw = np.asarray(dw).flatten()
    w = np.asarray(w).flatten()

    # Y_m (6×1)
    Y_m = np.vstack((dv, skew_np(p) @ dv))

    # Y_mp (6×3)
    Sdw = skew_np(dw)
    Sw = skew_np(w)
    Y_mp_t = Sdw + Sw @ Sw
    Y_mp_r = (skew_np(p) @ Sdw
              + skew_np(p) @ Sw @ Sw
              - skew_np(dv.flatten()))
    Y_mp = np.vstack((Y_mp_t, Y_mp_r))

    # Y_I (6×6) — top 3 rows zero, bottom 3 from Eq. 30 / Eq. 3.18
    dwx, dwy, dwz = dw
    wx, wy, wz = w
    Y_I = np.zeros((6, 6))
    Y_I[3:, :] = np.array([
        [dwx,        dwy - wx*wz,  dwz + wx*wy, -wy*wz,      wy**2 - wz**2, wy*wz],
        [wx*wz,      dwx + wy*wz, -wx**2 + wz**2, dwy,        dwz - wx*wy,  -wx*wz],
        [-wx*wy,     wx**2 - wy**2, dwx - wy*wz,  wx*wy,      dwy + wx*wz,   dwz],
    ])

    return np.hstack((Y_m, Y_mp, Y_I))
