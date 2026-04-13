"""Per-joint kinematics: symbolic transformation matrices and lambdified
Jacobians/derivatives used by both Newton-Euler and Euler-Lagrange methods."""
import numpy as np
import sympy

from .math_utils import (
    rpy_to_rotation_np, axis_rotation_sym, axis_torque_row,
    make_transform_np, skew_np,
)
from .urdf_parser import RobotDescription


class LinkKinematics:
    """Symbolic + lambdified kinematics for a single revolute joint/link."""

    def __init__(self, R0: np.ndarray, Rq_sym: sympy.Matrix,
                 di: np.ndarray, q_sym: sympy.Symbol):
        T_sym = self._build_transform(R0, Rq_sym, di)
        aJ_sym = self._angular_jacobian(T_sym, q_sym)
        pR_sym = sympy.Matrix(T_sym[:3, :3]).diff(q_sym)
        paJ_sym = aJ_sym.diff(q_sym)

        self.Ti_1_i = sympy.lambdify(q_sym, T_sym, modules="numpy")
        self.aJi_1_i = sympy.lambdify(q_sym, aJ_sym, modules="numpy")
        self.pRi_1_i = sympy.lambdify(q_sym, pR_sym, modules="numpy")
        self.paJi_1_i = sympy.lambdify(q_sym, paJ_sym, modules="numpy")

    @staticmethod
    def _build_transform(R0, Rq, di):
        """Build T_{i-1,i} with translation di = origin_xyz of this joint.

        di is the position of this joint's frame origin in the parent frame
        (constant, does not depend on q).
        """
        d = sympy.Matrix(di.flatten()).reshape(3, 1)
        R = sympy.Matrix(R0) * Rq
        top = R.row_join(d)
        return top.col_join(sympy.Matrix([[0, 0, 0, 1]]))

    @staticmethod
    def _angular_jacobian(T, q):
        R = sympy.Matrix(T[:3, :3])
        dR = R.diff(q)
        j1 = R[1, 0]*dR[2, 0] + R[1, 1]*dR[2, 1] + R[1, 2]*dR[2, 2]
        j2 = R[2, 0]*dR[0, 0] + R[2, 1]*dR[0, 1] + R[2, 2]*dR[0, 2]
        j3 = R[0, 0]*dR[1, 0] + R[0, 1]*dR[1, 1] + R[0, 2]*dR[1, 2]
        return sympy.Matrix([j1, j2, j3])


class RobotKinematics:
    """Kinematic model for the full serial chain, built from the URDF."""

    def __init__(self, robot: RobotDescription, logger=None):
        self.nDoF = robot.nDoF
        self.Tw_0 = robot.Tw_0.copy()
        self.link_kin = []        # list of LinkKinematics
        self.link_names = []      # child link names
        self.pi_i_raw = []        # position vector from this joint to the NEXT joint
        self.di_raw = []          # origin_xyz of each joint (position in parent frame)
        self.torque_row_sign = [] # (row_idx, sign) per joint
        self.pi_ci = []           # CoM xyz from inertial section
        self.Ri_ci = []           # rotation from inertial frame
        self.Ici_ci = []          # inertia tensor in CoM frame
        self.mass = []            # mass per link
        self.PI = np.zeros((0, 1))  # 10n x 1 full parameter vector

        # Build indexed chain from topology
        chain = robot.chain_joints
        # Extract only revolute joints in chain order, and the joint that follows each
        rev_chain = []  # list of (revolute_JointData, next_JointData_or_None)
        for idx, jd in enumerate(chain):
            if jd.joint_type in ("revolute", "continuous"):
                # Find the next joint in the chain (for offset to next frame)
                next_jd = None
                for k in range(idx + 1, len(chain)):
                    next_jd = chain[k]
                    if next_jd.joint_type in ("revolute", "continuous"):
                        break
                    elif next_jd.joint_type == "fixed":
                        # Accumulate fixed transforms between revolute joints
                        break
                else:
                    next_jd = None
                rev_chain.append((jd, idx))

        for count, (jd, chain_idx) in enumerate(rev_chain):
            q_sym = sympy.Symbol(f"q{count + 1}")
            link_name = jd.child

            # Compute offset to next joint: accumulate fixed transforms after
            # this revolute joint up to (and including) the next revolute joint
            pi_i = self._compute_next_joint_offset(chain, chain_idx)

            # di = position of THIS joint's frame in the parent link frame
            di = jd.origin_xyz.copy()

            R0 = rpy_to_rotation_np(*jd.origin_rpy)
            Rq = axis_rotation_sym(jd.axis, q_sym)
            row, sign = axis_torque_row(jd.axis)

            lk = LinkKinematics(R0, Rq, di, q_sym)
            self.link_kin.append(lk)
            self.link_names.append(link_name)
            self.pi_i_raw.append(pi_i)
            self.di_raw.append(di)
            self.torque_row_sign.append((row, sign))

            # Inertial parameters
            ld = robot.links.get(link_name)
            if ld is None:
                raise ValueError(f"Link '{link_name}' not found in URDF links.")
            m_i = ld.inertial.mass
            pi_ci = ld.inertial.origin_xyz.copy()
            rpy_ci = ld.inertial.origin_rpy
            R_ci = rpy_to_rotation_np(*rpy_ci)
            I_ci_local = ld.inertial.tensor

            # CoM offset relative to frame origin (joint position)
            pi_ci_i = pi_ci.reshape(3, 1)

            # Rotate inertia to link frame
            I_ci = R_ci @ I_ci_local @ R_ci.T

            # Parallel axis theorem -> inertia at frame origin (joint)
            S = skew_np(pi_ci_i)
            I_i = I_ci + m_i * (S.T @ S)

            self.pi_ci.append(pi_ci)
            self.Ri_ci.append(R_ci)
            self.Ici_ci.append(I_ci_local)
            self.mass.append(m_i)

            PI_i = np.array([
                m_i,
                m_i * pi_ci_i[0, 0], m_i * pi_ci_i[1, 0], m_i * pi_ci_i[2, 0],
                I_i[0, 0], I_i[0, 1], I_i[0, 2],
                I_i[1, 1], I_i[1, 2], I_i[2, 2],
            ]).reshape(10, 1)
            self.PI = np.vstack([self.PI, PI_i])

        if logger:
            logger.info("Built kinematics for %d DoF robot '%s'.", self.nDoF, robot.name)

    @staticmethod
    def _compute_next_joint_offset(chain, current_idx):
        """Walk forward from chain[current_idx] to find the next revolute joint.

        Returns the origin_xyz of that next joint. If there is no next revolute
        joint, returns zeros (terminal link).
        """
        for k in range(current_idx + 1, len(chain)):
            nj = chain[k]
            if nj.joint_type in ("revolute", "continuous"):
                return nj.origin_xyz.copy()
            # Could accumulate fixed-joint transforms here for complex chains,
            # but for simple serial chains the next revolute joint origin_xyz
            # already gives the position in the parent link frame.
        return np.zeros(3)

    def get_transform(self, q: np.ndarray, link_index: int) -> np.ndarray:
        """Return 4x4 world-to-link_index transform. *link_index* is 1-based."""
        T = self.Tw_0.copy()
        for i in range(link_index):
            T = T @ np.array(self.link_kin[i].Ti_1_i(q[i])).reshape(4, 4)
        return T
