"""Regression tests for sine-basis excitation initialisation and constraints."""
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP_ROOT = ROOT / "tmp_output"


def test_sine_basis_x0_drift_feasibility():
    from src.excitation import _build_literature_initial_guess
    from src.trajectory import build_frequencies, fourier_trajectory, param_bounds

    n_dof = 2
    m = 20
    f0 = 0.2
    freqs = build_frequencies(f0, m)
    q_lim = np.array([[-1.57, 1.57]] * n_dof)
    dq_lim = np.array([[-2.0, 2.0]] * n_dof)
    ddq_lim = np.array([[-5.0, 5.0]] * n_dof)
    q0 = np.zeros(n_dof)
    tf = 120.0 / f0
    dt = 1.0 / (2.0 * freqs[-1])
    t_init = np.arange(0.0, tf + dt, dt)

    bounds = param_bounds(
        n_dof, m, "sine", False, q_lim,
        freqs=freqs, dq_lim=dq_lim, ddq_lim=ddq_lim,
    )
    rng = np.random.default_rng(42)
    x0 = _build_literature_initial_guess(
        bounds, "sine", False, n_dof, m, freqs, q0, q_lim, dq_lim,
        ddq_lim, t_init, tf, rng,
    )

    t_check = np.linspace(0.0, tf, 5000)
    q_check, _, _ = fourier_trajectory(x0, freqs, t_check, q0, "sine", False)

    assert np.all(q_check >= q_lim[:, 0:1] - 1e-6), "x0 q lower bound violated"
    assert np.all(q_check <= q_lim[:, 1:2] + 1e-6), "x0 q upper bound violated"


def test_slsqp_constraints_sine_includes_lam1():
    from src.excitation import _build_slsqp_constraints

    n_dof = 2
    m = 5
    freqs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    q0 = np.zeros(n_dof)
    q_lim = np.array([[-1.57, 1.57]] * n_dof)
    dq_lim = np.array([[-2.0, 2.0]] * n_dof)
    ddq_lim = np.array([[-5.0, 5.0]] * n_dof)
    tf = 600.0
    t = np.linspace(0.0, tf, 200)

    cons = _build_slsqp_constraints(
        freqs, t, q0, "sine", True, q_lim, dq_lim, ddq_lim, n_dof, m=m, tf=tf,
    )

    assert len(cons) == 8 * n_dof


def test_slsqp_constraints_both_retains_lam1():
    from src.excitation import _build_slsqp_constraints

    n_dof = 2
    m = 5
    freqs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    q0 = np.zeros(n_dof)
    q_lim = np.array([[-1.57, 1.57]] * n_dof)
    dq_lim = np.array([[-2.0, 2.0]] * n_dof)
    ddq_lim = np.array([[-5.0, 5.0]] * n_dof)
    tf = 600.0
    t = np.linspace(0.0, tf, 200)

    cons = _build_slsqp_constraints(
        freqs, t, q0, "both", False, q_lim, dq_lim, ddq_lim, n_dof, m=m, tf=tf,
    )

    assert len(cons) == 8 * n_dof


def test_long_horizon_sine_basis_is_rejected_before_slsqp():
    from src.excitation import optimise_excitation

    kin = SimpleNamespace(nDoF=2)
    cfg_exc = {
        "basis_functions": "sine",
        "optimize_phase": True,
        "num_harmonics": 20,
        "base_frequency_hz": 0.2,
        "trajectory_duration_periods": 120,
        "optimizer_max_iter": 5,
        "optimize_condition_number": False,
        "torque_constraint_method": "none",
    }
    q_lim = np.array([[-1.57, 1.57], [-1.57, 1.57]])
    dq_lim = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    ddq_lim = np.array([[-5.0, 5.0], [-5.0, 5.0]])

    with pytest.raises(ValueError, match="fundamentally infeasible"):
        optimise_excitation(
            kin, cfg_exc, q_lim, dq_lim, ddq_lim,
            regressor_fn=lambda q, dq, ddq: np.zeros((kin.nDoF, 1)),
        )


def test_preflight_rejects_sine_with_optimize_phase_on_long_horizon():
    from src.excitation import preflight_excitation_config

    cfg_exc = {
        "basis_functions": "sine",
        "optimize_phase": True,
        "num_harmonics": 5,
        "base_frequency_hz": 0.2,
        "trajectory_duration_periods": 120,
        "torque_constraint_method": "none",
    }
    q_lim = np.array([[-1.57, 1.57], [-1.57, 1.57]])
    dq_lim = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    ddq_lim = np.array([[-1.5, 1.5], [-0.5, 0.5]])

    with pytest.raises(ValueError, match="fundamentally infeasible"):
        preflight_excitation_config(cfg_exc, q_lim, dq_lim, ddq_lim)


def test_preflight_both_basis_24_periods_reports_120s_duration():
    from src.excitation import preflight_excitation_config

    cfg_exc = {
        "basis_functions": "both",
        "optimize_phase": False,
        "num_harmonics": 20,
        "base_frequency_hz": 0.2,
        "trajectory_duration_periods": 24,
        "torque_constraint_method": "none",
    }
    q_lim = np.array([[-1.57, 1.57], [-1.57, 1.57]])
    dq_lim = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    ddq_lim = np.array([[-1.5, 1.5], [-0.5, 0.5]])

    assessment = preflight_excitation_config(cfg_exc, q_lim, dq_lim, ddq_lim)

    assert assessment["tf"] == pytest.approx(120.0)


def test_pipeline_passes_excitation_config_to_stage_5(tmp_path, monkeypatch):
    import json

    from src.config_loader import load_config
    from src.pipeline import SystemIdentificationPipeline

    cfg = {
        "urdf_path": str(ROOT / "tests" / "assets" / "RRBot_single.urdf"),
        "output_dir": str(tmp_path / "out"),
        "method": "newton_euler",
        "joint_limits": {
            "position": [[-1.57, 1.57], [-1.57, 1.57]],
            "velocity": [[-5.0, 5.0], [-5.0, 5.0]],
            "acceleration": [[-1.5, 1.5], [-0.5, 0.5]],
        },
        "excitation": {
            "basis_functions": "both",
            "optimize_phase": False,
            "num_harmonics": 20,
            "base_frequency_hz": 0.2,
            "trajectory_duration_periods": 24,
            "torque_constraint_method": "none",
        },
        "friction": {"model": "none"},
        "identification": {
            "solver": "ols",
            "parameter_bounds": False,
            "feasibility_method": "none",
            "data_file": None,
        },
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    loaded_cfg = load_config(str(cfg_path))

    captured = {}

    def stop_after_stage_5(kin, cfg_exc, q_lim, dq_lim, ddq_lim, **kwargs):
        captured["cfg_exc"] = deepcopy(cfg_exc)
        raise RuntimeError("STOP_AFTER_STAGE_5")

    monkeypatch.setattr("src.pipeline.optimise_excitation", stop_after_stage_5)

    pipe = SystemIdentificationPipeline(deepcopy(loaded_cfg))
    with pytest.raises(RuntimeError, match="STOP_AFTER_STAGE_5"):
        pipe.run()

    assert captured["cfg_exc"]["basis_functions"] == "both"
    assert captured["cfg_exc"]["optimize_phase"] is False
    assert captured["cfg_exc"]["trajectory_duration_periods"] == 24
