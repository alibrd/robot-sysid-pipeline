"""Slower comparison tests for the torque-limited excitation methods."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.test_torque_constraints import _run_pipeline, _write_torque_config, URDF_3DOF


@pytest.mark.slow
def test_slow_chance_monte_carlo_empirically_respects_reported_confidence():
    from src.torque_constraints import compute_torque_design_data

    def reg(q, dq, ddq):
        return np.array([[1.0, -0.5]])

    q = np.array([[0.0]])
    dq = np.array([[0.0]])
    ddq = np.array([[0.0]])
    params = np.array([4.0, 2.0])
    cfg = {
        "relative_stddev": 0.1,
        "absolute_stddev_floor": 0.0,
        "chance_confidence": 0.95,
    }
    tau_lim = np.array([[-10.0, 10.0]])
    design = compute_torque_design_data(q, dq, ddq, reg, params, tau_lim, "chance", cfg)
    sigma = np.array([0.4, 0.2])
    rng = np.random.default_rng(123)
    draws = params + rng.standard_normal((20000, 2)) * sigma
    tau_draws = np.array([reg(None, None, None) @ draw for draw in draws]).reshape(-1)
    inside = np.mean(
        (tau_draws >= design["design_lower"][0, 0]) &
        (tau_draws <= design["design_upper"][0, 0])
    )
    assert inside == pytest.approx(0.90, abs=0.01)


@pytest.mark.slow
def test_slow_all_six_methods_emit_comparable_summary_metrics(tmp_path):
    methods = [
        ("nominal_hard", {}),
        ("soft_penalty", {"soft_penalty_weight": 200.0, "soft_penalty_smoothing": 0.05}),
        ("robust_box", {"relative_uncertainty": 0.1, "absolute_uncertainty_floor": 1e-3}),
        ("chance", {"relative_stddev": 0.05, "absolute_stddev_floor": 1e-3, "chance_confidence": 0.95}),
        ("actuator_envelope", {"envelope_type": "constant"}),
        ("sequential_redesign", {"max_iterations": 2, "convergence_tol": 0.0}),
    ]
    seen = {}
    for method, torque_cfg in methods:
        cfg = _write_torque_config(
            tmp_path / method,
            torque_method=method,
            torque_constraint=torque_cfg,
            max_iter=10,
        )
        summary, _, _ = _run_pipeline(cfg)
        seen[method] = summary["max_identified_torque_ratio"]
        assert summary["torque_constraint_method"] == method
        assert summary["max_identified_torque_ratio"] is not None
    assert set(seen) == {method for method, _ in methods}


@pytest.mark.slow
@pytest.mark.parametrize(
    "torque_method,torque_constraint",
    [
        ("nominal_hard", {}),
        ("robust_box", {"relative_uncertainty": 0.1, "absolute_uncertainty_floor": 1e-3}),
    ],
)
def test_slow_strongest_two_methods_scale_to_3dof_fixture(tmp_path, torque_method, torque_constraint):
    cfg = _write_torque_config(
        tmp_path,
        urdf_path=URDF_3DOF,
        n_dof=3,
        torque_method=torque_method,
        torque_constraint=torque_constraint,
        torque_limits=[[-120.0, 120.0]] * 3,
        max_iter=10,
    )
    summary, torque, _ = _run_pipeline(cfg)
    assert summary["torque_identified_pass"] is True
    assert float(np.max(torque["identified_ratio"])) <= 1.0 + 1e-6


@pytest.mark.slow
def test_slow_oversampled_replay_detects_hidden_between_sample_violations():
    from src.torque_constraints import compute_torque_design_data
    from src.trajectory import build_frequencies, fourier_trajectory

    freqs = build_frequencies(0.2, 1)
    q0 = np.array([0.0])
    params = np.array([0.9])
    tau_lim = np.array([[-1.0, 1.0]])

    def reg(q, dq, ddq):
        return np.array([[1.0 + abs(float(q[0]))]])

    t_sparse = np.array([0.0, 5.0])
    q_s, dq_s, ddq_s = fourier_trajectory(params, freqs, t_sparse, q0, "cosine", False)
    sparse = compute_torque_design_data(q_s, dq_s, ddq_s, reg, np.array([0.5]), tau_lim, "nominal_hard", {})
    assert sparse["design_pass"] is True

    t_dense = np.linspace(0.0, 5.0, 101)
    q_d, dq_d, ddq_d = fourier_trajectory(params, freqs, t_dense, q0, "cosine", False)
    dense = compute_torque_design_data(q_d, dq_d, ddq_d, reg, np.array([0.5]), tau_lim, "nominal_hard", {})
    assert dense["design_pass"] is False
