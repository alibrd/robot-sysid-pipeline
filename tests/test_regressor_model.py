"""Tests for the first-class regressor model and exported artifacts."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URDF_RRBOT = ROOT / "tests" / "assets" / "RRBot_single.urdf"


def _minimal_cfg(tmp_path: Path, *, excitation_only=False, friction_model="none"):
    return {
        "urdf_path": str(URDF_RRBOT),
        "output_dir": str(tmp_path / "out"),
        "method": "newton_euler",
        "excitation_only": excitation_only,
        "joint_limits": {
            "position": [[-1.0, 1.0], [-1.0, 1.0]],
            "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
        },
        "excitation": {
            "basis_functions": "cosine",
            "optimize_phase": False,
            "num_harmonics": 1,
            "base_frequency_hz": 0.2,
            "optimize_condition_number": False,
            "optimizer_max_iter": 1,
            "trajectory_duration_periods": 1,
        },
        "friction": {"model": friction_model},
        "identification": {
            "solver": "ols",
            "parameter_bounds": False,
            "feasibility_method": "none",
            "data_file": None,
        },
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }


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


def test_excitation_only_writes_regressor_artifacts(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    pipe = SystemIdentificationPipeline(
        _minimal_cfg(tmp_path, excitation_only=True)
    )
    pipe.run()

    out = pipe.output_dir
    assert (out / "regressor_model.json").exists()
    assert (out / "regressor_model.urdf").exists()
    assert (out / "regressor_function.py").exists()
    assert (out / "checkpoint.npz").exists()
    assert (out / "excitation_trajectory.npz").exists()


def test_full_output_loads_regressor_and_multiplies_saved_parameter_vector(tmp_path):
    from src.pipeline import SystemIdentificationPipeline
    from src.regressor_model import load_regressor_model

    pipe = SystemIdentificationPipeline(
        _minimal_cfg(tmp_path, friction_model="viscous")
    )
    pipe.run()

    out = pipe.output_dir
    model = load_regressor_model(out / "regressor_model.json")
    results = np.load(str(out / "identification_results.npz"), allow_pickle=True)
    pi = np.asarray(results["pi_corrected"], dtype=float)

    q = np.array([0.15, -0.25])
    dq = np.array([0.2, -0.3])
    ddq = np.array([0.7, -0.8])
    tau = model.augmented(q, dq, ddq) @ pi

    assert tau.shape == (model.nDoF,)
    assert np.all(np.isfinite(tau))
    assert int(results["n_augmented_params"]) == pi.size

    summary = json.loads((out / "results_summary.json").read_text())
    assert summary["regressor_model"]["backend"] == "newton_euler"
    assert summary["regressor_model"]["n_augmented_params"] == pi.size


def test_exported_regressor_function_shim_imports(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    pipe = SystemIdentificationPipeline(
        _minimal_cfg(tmp_path, friction_model="viscous")
    )
    pipe.run()

    shim_path = pipe.output_dir / "regressor_function.py"
    spec = importlib.util.spec_from_file_location("regressor_function_test", shim_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    q = np.array([0.0, 0.1])
    dq = np.array([0.2, -0.2])
    ddq = np.array([0.3, -0.3])
    assert module.Y_rigid(q, dq, ddq).shape == (2, 20)
    assert module.Y_augmented(q, dq, ddq).shape == (2, 22)
    assert module.Y(q, dq, ddq).shape == (2, 22)
