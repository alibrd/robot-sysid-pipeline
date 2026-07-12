"""Per-joint kinematics: symbolic transformation matrices and lambdified
Jacobians/derivatives used by both Newton-Euler and Euler-Lagrange methods."""
import numpy as np
import sympy

from .math_utils import (
    rpy_to_rotation_np, axis_rotation_sym, axis_torque_row,
    skew_np, make_transform_np,
)
from .urdf_parser import RobotDescription


def _validate_revolute_axis(axis, joint_name: str) -> None:
    """Raise unless *axis* is (numerically) a signed coordinate-axis vector.

    The kinematics builder selects the rotation and the torque row via the
    dominant axis component, so a skew axis (e.g. [0.707, 0, 0.707]) would be
    silently projected onto one coordinate axis and every downstream quantity
    would be wrong.
    """
    ax = np.asarray(axis, dtype=float).flatten()
    norm = float(np.linalg.norm(ax))
    if ax.size != 3 or norm < 1e-12:
        raise ValueError(
            f"Joint '{joint_name}' has an invalid axis vector {ax.tolist()}."
        )
    if np.max(np.abs(ax / norm)) < 1.0 - 1e-9:
        raise ValueError(
            f"Joint '{joint_name}' axis {ax.tolist()} is not aligned with a "
            "coordinate axis. Only axes of the form ±[1,0,0], ±[0,1,0], "
            "±[0,0,1] (any scale) are supported; a skew axis would be "
            "silently projected onto its dominant component, producing wrong "
            "dynamics. Rotate the joint frame in the URDF (origin rpy) so "
            "the axis becomes axis-aligned."
        )


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

        # Retain symbolic forms for the standalone regressor export
        # (regressor_export.py walks them to emit numpy source).
        self.T_sym = T_sym
        self.aJ_sym = aJ_sym
        self.pR_sym = pR_sym
        self.paJ_sym = paJ_sym
        self.q_sym = q_sym

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


def iter_fixed_subtree(robot: RobotDescription, root_link_name: str,
                       logger=None):
    """Return ``[(link_name, T_root_link), ...]`` for *root_link_name* and
    every link attached to it through fixed-only joint paths.

    Public wrapper around the traversal used for inertia lumping, shared
    with the adapted-URDF exporter (which must zero exactly the inertials
    that were lumped).
    """
    children_of = {}
    for j in robot.joints:
        children_of.setdefault(j.parent, []).append(j)
    chain_joint_names = {jd.name for jd in robot.chain_joints}
    return RobotKinematics._fixed_subtree(
        root_link_name, children_of, chain_joint_names, logger
    )


class RobotKinematics:
    """Kinematic model for the full serial chain, built from the URDF."""

    def __init__(self, robot: RobotDescription, logger=None):
        self.nDoF = robot.nDoF
        self.Tw_0 = robot.Tw_0.copy()
        self.link_kin = []        # list of LinkKinematics
        self.link_names = []      # child link names
        self.pi_i_raw = []        # position of the NEXT revolute joint in this link frame
        self.di_raw = []          # effective joint origin (fixed transforms composed in)
        self.torque_row_sign = [] # (row_idx, sign) per joint
        self.pi_ci = []           # combined (lumped) CoM xyz in link frame
        self.Ri_ci = []           # identity: lumped tensor is expressed in link frame
        self.Ici_ci = []          # lumped inertia tensor about the combined CoM
        self.mass = []            # lumped mass per rigid body (link + fixed children)
        self.PI = np.zeros((0, 1))  # 10n x 1 full parameter vector

        # Build indexed chain from topology
        chain = robot.chain_joints
        unsupported = [
            (jd.name, jd.joint_type) for jd in chain
            if jd.joint_type not in ("fixed", "revolute", "continuous")
        ]
        if unsupported:
            raise NotImplementedError(
                "Only revolute/continuous (and fixed) joints are supported in "
                "the serial chain; treating other joint types as fixed would "
                "silently produce wrong dynamics. Unsupported joints: "
                + ", ".join(f"'{n}' ({t})" for n, t in unsupported)
            )
        rev_chain = [
            (jd, idx) for idx, jd in enumerate(chain)
            if jd.joint_type in ("revolute", "continuous")
        ]
        for jd, _ in rev_chain:
            _validate_revolute_axis(jd.axis, jd.name)

        # Effective per-revolute transforms: fixed joints BETWEEN two revolute
        # joints are composed into the following revolute joint's constant
        # pre-transform (R0_eff, d_eff), so T_{i-1,i}(q) = [R0_eff*Rq | d_eff].
        # Fixed joints BEFORE the first revolute joint are already folded into
        # Tw_0 by the parser.
        eff = []  # list of (R0_eff, d_eff) per revolute joint
        for count, (jd, chain_idx) in enumerate(rev_chain):
            R_acc = np.eye(3)
            t_acc = np.zeros(3)
            if count > 0:
                prev_idx = rev_chain[count - 1][1]
                for k in range(prev_idx + 1, chain_idx):
                    fj = chain[k]  # guaranteed fixed by the check above
                    t_acc = t_acc + R_acc @ fj.origin_xyz
                    R_acc = R_acc @ rpy_to_rotation_np(*fj.origin_rpy)
            R0_eff = R_acc @ rpy_to_rotation_np(*jd.origin_rpy)
            d_eff = t_acc + R_acc @ jd.origin_xyz
            eff.append((R0_eff, d_eff))

        # Map parent link -> child joints over ALL URDF joints (not just the
        # serial chain) so fixed subtrees hanging off any chain link are found.
        children_of = {}
        for j in robot.joints:
            children_of.setdefault(j.parent, []).append(j)
        chain_joint_names = {jd.name for jd in chain}

        for count, (jd, chain_idx) in enumerate(rev_chain):
            q_sym = sympy.Symbol(f"q{count + 1}")
            link_name = jd.child
            R0, di = eff[count]

            # Offset to the next revolute joint, expressed in this link frame
            # (the next joint's effective pre-transform translation).
            if count + 1 < len(rev_chain):
                pi_i = eff[count + 1][1].copy()
            else:
                pi_i = np.zeros(3)

            Rq = axis_rotation_sym(jd.axis, q_sym)
            row, sign = axis_torque_row(jd.axis)

            lk = LinkKinematics(R0, Rq, di, q_sym)
            self.link_kin.append(lk)
            self.link_names.append(link_name)
            self.pi_i_raw.append(pi_i)
            self.di_raw.append(di)
            self.torque_row_sign.append((row, sign))

            # Inertial parameters: lump this link plus every link attached to
            # it through fixed-only joint paths (hands, flanges, sensors, and
            # chain links between this and the next revolute joint). PyBullet
            # and other simulators merge fixed-child inertia into the parent;
            # the regressor must see the same rigid body.
            if link_name not in robot.links:
                raise ValueError(f"Link '{link_name}' not found in URDF links.")
            m_tot = 0.0
            h_tot = np.zeros(3)          # first moment m*c in link frame
            I_o_tot = np.zeros((3, 3))   # inertia at link-frame origin
            for sub_name, T_sub in self._fixed_subtree(
                    link_name, children_of, chain_joint_names, logger):
                ld = robot.links.get(sub_name)
                if ld is None:
                    continue  # joint child declared but no <link> inertial
                m_l = ld.inertial.mass
                R_sub = T_sub[:3, :3]
                t_sub = T_sub[:3, 3]
                R_cl = rpy_to_rotation_np(*ld.inertial.origin_rpy)
                # COM position and COM-frame inertia expressed in THIS link
                # frame (rotation only for the tensor: same reference point).
                c_i = t_sub + R_sub @ ld.inertial.origin_xyz
                I_c_i = (R_sub @ R_cl) @ ld.inertial.tensor @ (R_sub @ R_cl).T
                S = skew_np(c_i)
                m_tot += m_l
                h_tot += m_l * c_i
                I_o_tot += I_c_i + m_l * (S.T @ S)

            c_comb = h_tot / m_tot if m_tot > 0.0 else np.zeros(3)
            S_c = skew_np(c_comb)
            self.pi_ci.append(c_comb)
            self.Ri_ci.append(np.eye(3))
            self.Ici_ci.append(I_o_tot - m_tot * (S_c.T @ S_c))
            self.mass.append(m_tot)

            PI_i = np.array([
                m_tot,
                h_tot[0], h_tot[1], h_tot[2],
                I_o_tot[0, 0], I_o_tot[0, 1], I_o_tot[0, 2],
                I_o_tot[1, 1], I_o_tot[1, 2], I_o_tot[2, 2],
            ]).reshape(10, 1)
            self.PI = np.vstack([self.PI, PI_i])

        if logger:
            logger.info("Built kinematics for %d DoF robot '%s'.", self.nDoF, robot.name)

    @staticmethod
    def _fixed_subtree(root_link_name, children_of, chain_joint_names, logger):
        """Yield ``(link_name, T_root_link)`` for *root_link_name* and every
        link reachable from it through fixed joints only.

        Traversal stops at non-fixed joints: chain revolute joints are the
        next moving body (expected), while off-chain moving joints indicate a
        branch whose subtree cannot be modelled and triggers a warning.
        """
        out = [(root_link_name, np.eye(4))]
        stack = [(root_link_name, np.eye(4))]
        while stack:
            lname, T = stack.pop()
            for j in children_of.get(lname, []):
                if j.joint_type != "fixed":
                    if j.name not in chain_joint_names and logger:
                        logger.warning(
                            "Off-chain moving joint '%s' (%s) under link '%s' "
                            "is not part of the serial chain; its subtree "
                            "mass/inertia is EXCLUDED from the dynamics "
                            "model.", j.name, j.joint_type, lname,
                        )
                    continue
                T_child = T @ make_transform_np(
                    rpy_to_rotation_np(*j.origin_rpy), j.origin_xyz
                )
                out_entry = (j.child, T_child)
                out.append(out_entry)
                stack.append(out_entry)
        return out

    def get_transform(self, q: np.ndarray, link_index: int) -> np.ndarray:
        """Return 4x4 world-to-link_index transform. *link_index* is 1-based."""
        T = self.Tw_0.copy()
        for i in range(link_index):
            T = T @ np.array(self.link_kin[i].Ti_1_i(q[i])).reshape(4, 4)
        return T
