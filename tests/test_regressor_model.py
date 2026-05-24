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


