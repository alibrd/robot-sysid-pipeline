"""Tests for the first-class regressor model used internally."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URDF_RRBOT = ROOT / "tests" / "assets" / "RRBot_single.urdf"


def test_regressor_model_rigid_matches_newton_euler():
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.regressor_model import RegressorModel

    model = RegressorModel.from_urdf(URDF_RRBOT, backend="newton_euler")
    q = np.array([0.2, -0.3])
    dq = np.array([0.4, -0.1])
    ddq = np.array([1.2, -0.7])

    np.testing.assert_allclose(
        model.rigid(q, dq, ddq),
        newton_euler_regressor(model.kin, q, dq, ddq),
        atol=1e-12,
    )
    assert model.rigid(q, dq, ddq).shape == (model.nDoF, 10 * model.nDoF)


def test_regressor_model_augmented_appends_friction_block():
    from src.friction import build_friction_regressor
    from src.regressor_model import RegressorModel

    model = RegressorModel.from_urdf(
        URDF_RRBOT,
        friction_model="viscous_coulomb",
        backend="newton_euler",
    )
    q = np.array([0.1, -0.2])
    dq = np.array([0.3, -0.4])
    ddq = np.array([0.5, -0.6])

    expected = np.hstack((
        model.rigid(q, dq, ddq),
        build_friction_regressor(dq, "viscous_coulomb"),
    ))
    np.testing.assert_allclose(model.augmented(q, dq, ddq), expected, atol=1e-12)
    assert model.augmented(q, dq, ddq).shape[1] == 10 * model.nDoF + 3 * model.nDoF


def test_euler_lagrange_wrapper_returns_full_columns_and_matches_torque(tmp_path):
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.regressor_model import RegressorModel

    model = RegressorModel.from_urdf(
        URDF_RRBOT,
        backend="euler_lagrange",
        cache_dir=tmp_path / "el_cache",
    )
    pi = model.kin.PI.flatten()
    rng = np.random.default_rng(123)

    for _ in range(3):
        q = rng.uniform(-0.5, 0.5, model.nDoF)
        dq = rng.uniform(-1.0, 1.0, model.nDoF)
        ddq = rng.uniform(-2.0, 2.0, model.nDoF)
        Y_el = model.rigid(q, dq, ddq)
        assert Y_el.shape == (model.nDoF, 10 * model.nDoF)
        np.testing.assert_allclose(
            Y_el @ pi,
            newton_euler_regressor(model.kin, q, dq, ddq) @ pi,
            atol=1e-10,
        )




_URDF_1DOF_TEMPLATE = """<?xml version="1.0"?>
<robot name="cache_test">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0.1"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="{origin}"/>
    <axis xyz="{axis}"/>
    <limit lower="-1" upper="1" velocity="2" effort="10"/>
  </joint>
</robot>
"""


def test_el_cache_rebuilds_when_urdf_geometry_changes(tmp_path):
    """Two geometrically different robots sharing one cache directory must
    NOT share the cached symbolic regressor (fingerprint invalidation)."""
    import numpy as np

    from src.regressor_model import RegressorModel

    urdf_a = tmp_path / "robot_a.urdf"
    urdf_a.write_text(
        _URDF_1DOF_TEMPLATE.format(origin="0 0 0.2", axis="0 0 1"),
        encoding="utf-8",
    )
    urdf_b = tmp_path / "robot_b.urdf"
    urdf_b.write_text(
        _URDF_1DOF_TEMPLATE.format(origin="0.1 0 0.5", axis="0 1 0"),
        encoding="utf-8",
    )
    shared_cache = tmp_path / "shared_el_cache"

    model_a = RegressorModel.from_urdf(
        urdf_a, backend="euler_lagrange", cache_dir=shared_cache
    )
    # Same cache dir, different geometry: a stale cache would silently
    # return robot A's regressor for robot B.
    model_b_el = RegressorModel.from_urdf(
        urdf_b, backend="euler_lagrange", cache_dir=shared_cache
    )
    model_b_ne = RegressorModel.from_urdf(urdf_b, backend="newton_euler")

    pi_b = model_b_ne.kin.PI.flatten()
    rng = np.random.default_rng(7)
    for _ in range(3):
        q = rng.uniform(-1, 1, 1)
        dq = rng.uniform(-2, 2, 1)
        ddq = rng.uniform(-4, 4, 1)
        tau_el = model_b_el.rigid(q, dq, ddq) @ pi_b
        tau_ne = model_b_ne.rigid(q, dq, ddq) @ pi_b
        np.testing.assert_allclose(tau_el, tau_ne, atol=1e-9)
    # Sanity: A and B really are different robots.
    q = np.array([0.4])
    dq = np.array([0.6])
    ddq = np.array([1.2])
    tau_a = model_a.rigid(q, dq, ddq) @ model_a.kin.PI.flatten()
    tau_b = model_b_el.rigid(q, dq, ddq) @ pi_b
    assert abs(float(tau_a[0] - tau_b[0])) > 1e-3
