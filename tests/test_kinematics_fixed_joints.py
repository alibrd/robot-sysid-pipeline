"""Regression tests for fixed-joint handling in the kinematic chain.

Two properties are enforced:

1. Fixed joints BETWEEN revolute joints are composed (translation AND
   rotation) into the following revolute joint's constant pre-transform.
2. Links attached through fixed-only joint paths (hands, flanges, sensors,
   COM-marker links) are lumped into the preceding revolute body: masses,
   first moments, and origin-referred inertia tensors add up, exactly as
   PyBullet merges fixed-child inertia into the parent body.

Before these fixes the pipeline silently dropped both, which is why the
Drake pendulum (0.5 kg on a fixed-welded ``arm_com`` link) and RRBot
(1e-5 kg sensor links) disagreed with PyBullet inverse dynamics.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_PENDULUM = str(ASSET_DIR / "DrakePendulum_1DoF.urdf")
URDF_RRBOT = str(ASSET_DIR / "RRBot_single.urdf")


def _build_kin(urdf_path):
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf
    return RobotKinematics(parse_urdf(str(urdf_path)))


# ---------------------------------------------------------------------------
# 1. Inertia lumping
# ---------------------------------------------------------------------------

def test_drake_pendulum_fixed_child_mass_is_lumped():
    """arm (0.5 kg) + fixed-welded arm_com (0.5 kg), both at z=-0.5."""
    kin = _build_kin(URDF_PENDULUM)
    assert kin.nDoF == 1
    np.testing.assert_allclose(kin.mass, [1.0], atol=1e-12)
    pi = kin.PI.flatten()
    # [m, m*cx, m*cy, m*cz, ...] with combined COM at (0, 0, -0.5)
    np.testing.assert_allclose(pi[:4], [1.0, 0.0, 0.0, -0.5], atol=1e-12)


def test_rrbot_terminal_sensor_links_are_lumped():
    """link3 (1 kg) + hokuyo (1e-5 kg) + camera (1e-5 kg) via fixed joints."""
    kin = _build_kin(URDF_RRBOT)
    assert kin.nDoF == 2
    np.testing.assert_allclose(kin.mass[1], 1.0 + 2e-5, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Mid-chain fixed-joint transform composition
# ---------------------------------------------------------------------------

_URDF_WITH_FIXED = """<?xml version="1.0"?>
<robot name="midfix">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0.25"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="interlink"/>
  <link name="link2">
    <inertial>
      <origin xyz="0.1 0 0.2" rpy="0 0.2 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0.001" ixz="0" iyy="0.012" iyz="0" izz="0.008"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/>
  </joint>
  <joint name="weld" type="fixed">
    <parent link="link1"/><child link="interlink"/>
    <origin xyz="0.1 0 0.2" rpy="0.3 0 0"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="interlink"/><child link="link2"/>
    <origin xyz="0 0.05 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/>
  </joint>
</robot>
"""

_URDF_COMPOSED = """<?xml version="1.0"?>
<robot name="midfix_composed">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0.25"/>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <origin xyz="0.1 0 0.2" rpy="0 0.2 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.015" ixy="0.001" ixz="0" iyy="0.012" iyz="0" izz="0.008"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="{xyz}" rpy="0.3 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" velocity="2" effort="50"/>
  </joint>
</robot>
"""


def test_mid_chain_fixed_joint_transform_is_composed(tmp_path):
    """A rotated+translated fixed joint between two revolutes must be
    equivalent to a URDF where the transform is pre-composed by hand."""
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.math_utils import rot_x_np

    # Compose by hand: T_fixed = [rot_x(0.3) | (0.1, 0, 0.2)], then j2's
    # origin (0, 0.05, 0.3) expressed through it.
    R_f = rot_x_np(0.3)
    xyz = np.array([0.1, 0.0, 0.2]) + R_f @ np.array([0.0, 0.05, 0.3])
    xyz_str = " ".join(f"{v:.17g}" for v in xyz)

    urdf_a = tmp_path / "with_fixed.urdf"
    urdf_a.write_text(_URDF_WITH_FIXED, encoding="utf-8")
    urdf_b = tmp_path / "composed.urdf"
    urdf_b.write_text(_URDF_COMPOSED.format(xyz=xyz_str), encoding="utf-8")

    kin_a = _build_kin(urdf_a)
    kin_b = _build_kin(urdf_b)

    assert kin_a.nDoF == kin_b.nDoF == 2
    np.testing.assert_allclose(kin_a.PI, kin_b.PI, atol=1e-12)

    rng = np.random.default_rng(4)
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, 2)
        dq = rng.uniform(-2, 2, 2)
        ddq = rng.uniform(-4, 4, 2)
        Y_a = newton_euler_regressor(kin_a, q, dq, ddq)
        Y_b = newton_euler_regressor(kin_b, q, dq, ddq)
        np.testing.assert_allclose(Y_a, Y_b, atol=1e-10)


def test_mid_chain_fixed_joint_el_matches_ne(tmp_path):
    """The Euler-Lagrange backend sees the same composed transforms."""
    from src.regressor_model import RegressorModel

    urdf_a = tmp_path / "with_fixed.urdf"
    urdf_a.write_text(_URDF_WITH_FIXED, encoding="utf-8")

    model_ne = RegressorModel.from_urdf(urdf_a, backend="newton_euler")
    model_el = RegressorModel.from_urdf(
        urdf_a, backend="euler_lagrange", cache_dir=tmp_path / "el_cache"
    )
    pi = model_ne.kin.PI.flatten()

    rng = np.random.default_rng(5)
    for _ in range(3):
        q = rng.uniform(-1.5, 1.5, 2)
        dq = rng.uniform(-2, 2, 2)
        ddq = rng.uniform(-4, 4, 2)
        tau_ne = model_ne.rigid(q, dq, ddq) @ pi
        tau_el = model_el.rigid(q, dq, ddq) @ pi
        np.testing.assert_allclose(tau_ne, tau_el, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. End-to-end acceptance: match PyBullet inverse dynamics
# ---------------------------------------------------------------------------

@pytest.mark.skipif(importlib.util.find_spec("pybullet") is None,
                    reason="pybullet not installed")
@pytest.mark.parametrize("urdf_path", [URDF_PENDULUM, URDF_RRBOT])
def test_fixed_subtree_dynamics_match_pybullet(urdf_path):
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.pybullet_validation import compute_torques
    from src.urdf_parser import parse_urdf
    from src.kinematics import RobotKinematics

    robot = parse_urdf(urdf_path)
    kin = RobotKinematics(robot)
    pi = kin.PI.flatten()
    rng = np.random.default_rng(0)
    n = kin.nDoF
    N = 20
    q = rng.uniform(-1.5, 1.5, (N, n))
    dq = rng.uniform(-2, 2, (N, n))
    ddq = rng.uniform(-4, 4, (N, n))

    tau_pipe = np.array([
        newton_euler_regressor(kin, q[k], dq[k], ddq[k]) @ pi for k in range(N)
    ])
    tau_pb = compute_torques(
        urdf_path, robot.revolute_joint_names, q, dq, ddq, [0, 0, -9.80665]
    )
    assert np.max(np.abs(tau_pipe - tau_pb)) < 1e-9
