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
        "constraint_style": "literature_standard",
        "trajectory_duration_periods": 120,
        "optimizer_max_iter": 5,
        "optimizer_pop_size": 4,
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


def test_pipeline_preflight_rejects_long_horizon_sine_before_optimization(monkeypatch):
    from src.config_loader import load_config
    from src.pipeline import SystemIdentificationPipeline

    cfg = load_config(str(ROOT / "config" / "rrbot_single_2min_20harm_pipeline.json"))
    cfg["excitation"]["basis_functions"] = "sine"
    cfg["excitation"]["optimize_phase"] = True
    cfg["excitation"]["trajectory_duration_periods"] = 120

    def fail_if_called(*args, **kwargs):
        raise AssertionError("optimise_excitation should not run after preflight failure")

    monkeypatch.setattr("src.pipeline.optimise_excitation", fail_if_called)

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(TMP_ROOT / "pytest_preflight_reject" / "out")
    pipe = SystemIdentificationPipeline(deepcopy(cfg))
    with pytest.raises(ValueError, match="fundamentally infeasible"):
        pipe.run()


def test_rrbot_single_2min_20harm_sample_config_matches_supported_setup():
    from src.config_loader import load_config
    from src.excitation import preflight_excitation_config

    cfg = load_config(str(ROOT / "config" / "rrbot_single_2min_20harm_pipeline.json"))
    q_lim = np.asarray(cfg["joint_limits"]["position"], dtype=float)
    dq_lim = np.asarray(cfg["joint_limits"]["velocity"], dtype=float)
    ddq_lim = np.asarray(cfg["joint_limits"]["acceleration"], dtype=float)

    assessment = preflight_excitation_config(cfg["excitation"], q_lim, dq_lim, ddq_lim)

    assert cfg["excitation"]["basis_functions"] == "both"
    assert cfg["excitation"]["optimize_phase"] is False
    assert cfg["excitation"]["trajectory_duration_periods"] == 24
    assert assessment["tf"] == pytest.approx(120.0)


def test_pipeline_reaches_stage_5_for_rrbot_single_2min_20harm_sample(monkeypatch):
    from src.config_loader import load_config
    from src.pipeline import SystemIdentificationPipeline

    cfg = load_config(str(ROOT / "config" / "rrbot_single_2min_20harm_pipeline.json"))
    captured = {}

    def stop_after_stage_5(kin, cfg_exc, q_lim, dq_lim, ddq_lim, **kwargs):
        captured["cfg_exc"] = deepcopy(cfg_exc)
        raise RuntimeError("STOP_AFTER_STAGE_5")

    monkeypatch.setattr("src.pipeline.optimise_excitation", stop_after_stage_5)

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(TMP_ROOT / "pytest_stage5_sample" / "out")
    pipe = SystemIdentificationPipeline(deepcopy(cfg))
    with pytest.raises(RuntimeError, match="STOP_AFTER_STAGE_5"):
        pipe.run()

    assert captured["cfg_exc"]["basis_functions"] == "both"
    assert captured["cfg_exc"]["optimize_phase"] is False
    assert captured["cfg_exc"]["trajectory_duration_periods"] == 24
