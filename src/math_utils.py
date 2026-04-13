"""Mathematical utility functions: rotations, skew-symmetric matrices, constants."""
import numpy as np
import sympy

GRAVITY_SI = 9.80665
GRAVITY = np.array([0.0, 0.0, -GRAVITY_SI])


# ── Numeric (NumPy) rotation matrices ──────────────────────────────────────

def rot_x_np(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def rot_y_np(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])


def rot_z_np(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def rpy_to_rotation_np(r, p, y):
    return rot_z_np(y) @ rot_y_np(p) @ rot_x_np(r)


# ── Symbolic (SymPy) rotation matrices ─────────────────────────────────────

def rot_x_sym(q):
    return sympy.Matrix([
        [1, 0, 0],
        [0, sympy.cos(q), -sympy.sin(q)],
        [0, sympy.sin(q), sympy.cos(q)]
    ])


def rot_y_sym(q):
    return sympy.Matrix([
        [sympy.cos(q), 0, sympy.sin(q)],
        [0, 1, 0],
        [-sympy.sin(q), 0, sympy.cos(q)]
    ])


def rot_z_sym(q):
    return sympy.Matrix([
        [sympy.cos(q), -sympy.sin(q), 0],
        [sympy.sin(q), sympy.cos(q), 0],
        [0, 0, 1]
    ])


def axis_rotation_sym(axis, q):
    """Symbolic rotation about *axis* ([1,0,0], [0,-1,0], etc.) by angle *q*."""
    ax = np.array(axis, dtype=float)
    idx = int(np.argmax(np.abs(ax)))
    sign = float(np.sign(ax[idx]))
    angle = q * sign
    return [rot_x_sym, rot_y_sym, rot_z_sym][idx](angle)


def axis_torque_row(axis):
    """Return (row_index, sign) for selecting the torque row from a 6-vector
    [fx, fy, fz, tx, ty, tz] corresponding to the joint *axis*.
    """
    ax = np.array(axis, dtype=float)
    idx = int(np.argmax(np.abs(ax)))
    sign = float(np.sign(ax[idx]))
    return idx + 3, sign


# ── Skew-symmetric matrices ───────────────────────────────────────────────

def skew_np(v):
    v = np.asarray(v).flatten()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def skew_sym(v):
    return sympy.Matrix([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])


# ── Transformation matrix helpers ──────────────────────────────────────────

def make_transform_np(R, p):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R)
    T[:3, 3] = np.asarray(p).flatten()
    return T


def make_transform_sym(R, p):
    p = sympy.Matrix(p).reshape(3, 1)
    top = sympy.Matrix(R).row_join(p)
    return top.col_join(sympy.Matrix([[0, 0, 0, 1]]))


def get_rotation_np(T):
    return np.array(T[:3, :3])


def get_translation_np(T):
    return np.array(T[:3, 3])


def get_rotation_sym(T):
    return sympy.Matrix(T)[:3, :3]


def get_translation_sym(T):
    return sympy.Matrix(T)[:3, 3]
