"""Torque-limited excitation tests covering config, math helpers, and end-to-end runs."""
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_RRBOT = str(ASSET_DIR / "RRBot_single.urdf")
URDF_1DOF = str(ASSET_DIR / "SC_1DoF.urdf")
URDF_3DOF = str(ASSET_DIR / "SC_3DoF.urdf")


def _write_torque_config(tmp_path,
                         *,
                         urdf_path=URDF_RRBOT,
                         n_dof=2,
                         method="newton_euler",
                         torque_method="nominal_hard",
                         torque_limits=None,
                         torque_constraint=None,
                         optimize_condition_number=False,
                         num_harmonics=1,
                         max_iter=20,
                         feasibility="none"):
    torque_limits = torque_limits or [[-50.0, 50.0]] * n_dof
    torque_constraint = torque_constraint or {}
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "urdf_path": urdf_path,
        "output_dir": str(tmp_path / "out"),
        "method": method,
        "joint_limits": {
            "position": [[-1.0, 1.0]] * n_dof,
            "velocity": [[-2.0, 2.0]] * n_dof,
            "acceleration": [[-5.0, 5.0]] * n_dof,
            "torque": torque_limits,
        },
        "excitation": {
            "basis_functions": "cosine",
            "optimize_phase": False,
            "num_harmonics": num_harmonics,
            "base_frequency_hz": 0.2,
            "optimize_condition_number": optimize_condition_number,
            "optimizer_max_iter": max_iter,
            "trajectory_duration_periods": 1,
            "torque_constraint_method": torque_method,
            "torque_validation_oversample_factor": 5,
            "torque_constraint": torque_constraint,
        },
        "friction": {"model": "none"},
        "identification": {
            "solver": "ols",
            "parameter_bounds": False,
            "feasibility_method": feasibility,
            "data_file": None,
        },
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }
    cfg_path = tmp_path / f"{torque_method}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _run_pipeline(cfg_path):
    from src.pipeline import SystemIdentificationPipeline

    pipe = SystemIdentificationPipeline(str(cfg_path))
    pipe.run()
    out_dir = Path(pipe.cfg["output_dir"])
    summary = json.loads((out_dir / "results_summary.json").read_text(encoding="utf-8"))
    torque = np.load(out_dir / "torque_limit_validation.npz", allow_pickle=True)
    results = np.load(out_dir / "identification_results.npz", allow_pickle=True)
    return summary, torque, results


def test_stage_0_config_accepts_torque_fields_and_rejects_invalid_combinations(tmp_path):
    from src.config_loader import load_config

    good = _write_torque_config(tmp_path / "good")
    cfg = load_config(str(good))
    assert cfg["joint_limits"]["torque"][0] == [-50.0, 50.0]
    assert cfg["excitation"]["torque_constraint_method"] == "nominal_hard"
    assert "constraint_style" not in cfg["excitation"]
    assert "optimizer_pop_size" not in cfg["excitation"]

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({
        "urdf_path": URDF_RRBOT,
        "output_dir": str(tmp_path / "legacy_out"),
        "joint_limits": {
            "position": [[-1.0, 1.0], [-1.0, 1.0]],
            "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
            "torque": [[-50.0, 50.0], [-50.0, 50.0]],
        },
        "excitation": {
            "constraint_style": "legacy_excTrajGen",
            "optimizer_pop_size": 99,
            "torque_constraint_method": "nominal_hard",
        },
    }), encoding="utf-8")
    legacy_cfg = load_config(str(legacy))
    assert legacy_cfg["excitation"]["torque_constraint_method"] == "nominal_hard"
    assert "constraint_style" not in legacy_cfg["excitation"]
    assert "optimizer_pop_size" not in legacy_cfg["excitation"]

    bad_seq = tmp_path / "bad_seq.json"
    bad_seq.write_text(json.dumps({
        "urdf_path": URDF_RRBOT,
        "output_dir": str(tmp_path / "bad_seq_out"),
        "method": "euler_lagrange",
        "excitation": {
            "torque_constraint_method": "sequential_redesign",
        },
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="sequential_redesign"):
        load_config(str(bad_seq))

    bad_chance = tmp_path / "bad_chance.json"
    bad_chance.write_text(json.dumps({
        "urdf_path": URDF_RRBOT,
        "output_dir": str(tmp_path / "bad_chance_out"),
        "joint_limits": {
            "position": [[-1.0, 1.0], [-1.0, 1.0]],
            "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
            "torque": [[-50.0, 50.0], [-50.0, 50.0]],
        },
        "excitation": {
            "torque_constraint_method": "chance",
            "torque_constraint": {"chance_confidence": 1.0},
        },
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="chance_confidence"):
        load_config(str(bad_chance))


_EFFORT_URDF = """\
<?xml version="1.0" ?>
<robot name="TestBot">
  <link name="base"/>
  <joint name="Joint_1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="12.0" velocity="2.5" lower="-1.2" upper="1.2"/>
  </joint>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
"""


def test_urdf_effort_limit_is_parsed_from_joint_element(tmp_path):
    from src.urdf_parser import parse_urdf

    urdf_path = tmp_path / "effort_bot.urdf"
    urdf_path.write_text(_EFFORT_URDF, encoding="utf-8")

    robot = parse_urdf(str(urdf_path))
    joint = next(j for j in robot.joints if j.name == "Joint_1")
    assert joint.limit_effort == 12.0
    assert joint.limit_velocity == 2.5
    assert joint.limit_lower == -1.2
    assert joint.limit_upper == 1.2


def test_torque_limit_precedence_urdf_over_json_then_json_fallback_then_error(tmp_path):
    from src.urdf_parser import extract_torque_limits, parse_urdf

    urdf_path = tmp_path / "effort_bot.urdf"
    urdf_path.write_text(_EFFORT_URDF, encoding="utf-8")
    _logger = type("L", (), {"debug": lambda *a, **kw: None})()

    robot_effort = parse_urdf(str(urdf_path))
    tau_lim, sources = extract_torque_limits(
        robot_effort, {"torque": [[-99.0, 99.0]]}, logger=_logger, required=True,
    )
    np.testing.assert_allclose(tau_lim, [[-12.0, 12.0]])
    assert sources == ["urdf_effort"]

    robot_no_effort = parse_urdf(URDF_1DOF)
    tau_lim, sources = extract_torque_limits(
        robot_no_effort, {"torque": [[-15.0, 15.0]]}, logger=_logger, required=True,
    )
    np.testing.assert_allclose(tau_lim, [[-15.0, 15.0]])
    assert sources == ["json_torque"]

    with pytest.raises(ValueError, match="Torque limits missing"):
        extract_torque_limits(
            robot_no_effort, {"torque": None}, logger=_logger, required=True,
        )


@pytest.mark.parametrize("method", ["newton_euler", "euler_lagrange"])
def test_stage_3_to_5_torque_evaluator_matches_direct_regressor_for_ne_and_el(tmp_path, method):
    from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.torque_constraints import (
        build_nominal_parameter_vector,
        evaluate_torque_series,
        make_augmented_regressor,
    )
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_RRBOT))
    if method == "newton_euler":
        base_reg = lambda q, dq, ddq: newton_euler_regressor(kin, q, dq, ddq)
        rigid = kin.PI.flatten()
    else:
        reg_el, kept = euler_lagrange_regressor_builder(kin, str(tmp_path / "el_cache"))
        base_reg = reg_el
        rigid = kin.PI.flatten()[kept]

    aug_reg = make_augmented_regressor(base_reg, "none")
    params = build_nominal_parameter_vector(rigid, kin.nDoF, "none")
    q = np.array([[0.1, -0.2], [0.05, 0.15]])
    dq = np.array([[0.3, -0.4], [0.1, -0.2]])
    ddq = np.array([[0.5, -0.3], [-0.1, 0.2]])
    tau_series, _ = evaluate_torque_series(q, dq, ddq, aug_reg, params)
    manual = np.column_stack([
        aug_reg(q[:, idx], dq[:, idx], ddq[:, idx]) @ params for idx in range(q.shape[1])
    ])
    np.testing.assert_allclose(tau_series, manual, atol=1e-12)


def test_stage_6_torque_design_dispatch_produces_expected_method_specific_fields():
    from src.torque_constraints import compute_torque_design_data

    def reg(q, dq, ddq):
        return np.array([[1.0, 2.0]])

    q = np.array([[0.0, 0.0]])
    dq = np.array([[0.0, 0.5]])
    ddq = np.array([[0.0, 0.0]])
    params = np.array([2.0, 1.0])
    tau_lim = np.array([[-10.0, 10.0]])

    nominal = compute_torque_design_data(q, dq, ddq, reg, params, tau_lim, "nominal_hard", {})
    robust = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "robust_box",
        {"relative_uncertainty": 0.1, "absolute_uncertainty_floor": 0.0},
    )
    chance = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "chance",
        {"relative_stddev": 0.2, "absolute_stddev_floor": 0.0, "chance_confidence": 0.95},
    )
    envelope = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "actuator_envelope",
        {"envelope_type": "speed_linear", "speed_linear_slope": 0.5, "velocity_reference": 1.0,
         "min_scale": 0.5, "max_scale": 1.0},
    )

    np.testing.assert_allclose(nominal["design_upper"], nominal["tau_nominal"])
    assert np.all(robust["design_upper"] >= nominal["design_upper"])
    assert np.all(robust["design_lower"] <= nominal["design_lower"])
    assert chance["quantile"] == pytest.approx(NormalDist().inv_cdf(0.95))
    assert envelope["limit_upper"][0, 1] < envelope["limit_upper"][0, 0]


def test_stage_8_observation_matrix_is_unchanged_by_torque_config():
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.observation_matrix import build_observation_matrix
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_RRBOT))
    rng = np.random.default_rng(7)
    q = rng.uniform(-0.3, 0.3, size=(12, kin.nDoF))
    dq = rng.uniform(-0.4, 0.4, size=(12, kin.nDoF))
    ddq = rng.uniform(-0.5, 0.5, size=(12, kin.nDoF))
    tau = rng.uniform(-1.0, 1.0, size=(12, kin.nDoF))
    reg = lambda qv, dqv, ddqv: newton_euler_regressor(kin, qv, dqv, ddqv)
    cfg_plain = {
        "friction": {"model": "none"},
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
        "excitation": {"torque_constraint_method": "none"},
    }
    cfg_torque = {
        "friction": {"model": "none"},
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
        "excitation": {"torque_constraint_method": "nominal_hard"},
    }
    W_plain, tau_plain = build_observation_matrix(q, dq, ddq, tau, reg, cfg_plain, 100.0)
    W_torque, tau_torque = build_observation_matrix(q, dq, ddq, tau, reg, cfg_torque, 100.0)
    np.testing.assert_allclose(W_plain, W_torque, atol=1e-12)
    np.testing.assert_allclose(tau_plain, tau_torque, atol=1e-12)


def test_nominal_hard_unit_constraints_evaluate_expected_sign_and_value():
    from src.torque_constraints import compute_torque_design_data

    def reg(q, dq, ddq):
        return np.array([[1.0]])

    q = np.array([[0.0, 0.0]])
    dq = np.array([[0.0, 0.0]])
    ddq = np.array([[0.0, 0.0]])
    params = np.array([6.0])
    tau_lim = np.array([[-5.0, 5.0]])
    design = compute_torque_design_data(q, dq, ddq, reg, params, tau_lim, "nominal_hard", {})
    assert not design["design_pass"]
    assert design["design_upper_margin"][0, 0] == pytest.approx(-1.0)
    assert design["design_lower_margin"][0, 0] == pytest.approx(11.0)


def test_hard_torque_constraints_apply_default_optimization_guard_band(monkeypatch):
    from src.excitation import _build_torque_constraints

    captured = {}

    def fake_compute_torque_design_data(
        q, dq, ddq, get_regressor_fn, nominal_params, tau_lim, method, torque_cfg
    ):
        captured["tau_lim"] = np.array(tau_lim, copy=True)
        n_samples = q.shape[1]
        return {
            "tau_nominal": np.zeros((1, n_samples)),
            "limit_lower": np.full((1, n_samples), tau_lim[0, 0]),
            "limit_upper": np.full((1, n_samples), tau_lim[0, 1]),
            "design_lower": np.zeros((1, n_samples)),
            "design_upper": np.zeros((1, n_samples)),
        }

    monkeypatch.setattr("src.excitation.compute_torque_design_data", fake_compute_torque_design_data)

    constraints = _build_torque_constraints(
        freqs=np.array([0.5]),
        t=np.array([0.0, 1.0]),
        q0=np.array([0.0]),
        basis="cosine",
        opt_phase=False,
        nDoF=1,
        torque_method="nominal_hard",
        tau_lim=np.array([[-10.0, 10.0]]),
        nominal_params=np.array([1.0]),
        get_regressor_fn=lambda q, dq, ddq: np.array([[1.0]]),
        torque_cfg={},
    )

    constraints[0]["fun"](np.array([0.0]))

    np.testing.assert_allclose(captured["tau_lim"], [[-9.8, 9.8]])


def test_soft_penalty_unit_is_zero_inside_bounds_and_increases_with_violation():
    from src.torque_constraints import compute_soft_penalty

    lower = np.array([[-5.0, -5.0]])
    upper = np.array([[5.0, 5.0]])
    inside = compute_soft_penalty(np.array([[0.5, -0.5]]), lower, upper, 10.0, 0.1)
    mild = compute_soft_penalty(np.array([[5.5, -5.5]]), lower, upper, 10.0, 0.1)
    severe = compute_soft_penalty(np.array([[8.0, -8.0]]), lower, upper, 10.0, 0.1)
    assert inside < 1e-8
    assert mild > inside
    assert severe > mild


def test_robust_box_unit_matches_bruteforce_corner_enumeration():
    from src.torque_constraints import compute_torque_design_data

    def reg(q, dq, ddq):
        return np.array([[2.0, -1.0]])

    q = np.array([[0.0]])
    dq = np.array([[0.0]])
    ddq = np.array([[0.0]])
    params = np.array([3.0, 4.0])
    tau_lim = np.array([[-50.0, 50.0]])
    design = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "robust_box",
        {"relative_uncertainty": 0.1, "absolute_uncertainty_floor": 0.0},
    )
    delta = np.array([0.3, 0.4])
    corners = []
    for s0 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            corners.append(reg(None, None, None) @ (params + np.array([s0, s1]) * delta))
    corners = np.array(corners).reshape(-1)
    assert design["design_upper"][0, 0] == pytest.approx(np.max(corners))
    assert design["design_lower"][0, 0] == pytest.approx(np.min(corners))


def test_chance_unit_matches_manual_gaussian_margin():
    from src.torque_constraints import compute_torque_design_data

    def reg(q, dq, ddq):
        return np.array([[3.0, 4.0]])

    q = np.array([[0.0]])
    dq = np.array([[0.0]])
    ddq = np.array([[0.0]])
    params = np.array([2.0, 1.0])
    tau_lim = np.array([[-50.0, 50.0]])
    design = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "chance",
        {"relative_stddev": 0.1, "absolute_stddev_floor": 0.0, "chance_confidence": 0.9},
    )
    z = NormalDist().inv_cdf(0.9)
    sigma = np.array([0.2, 0.1])
    manual_margin = z * np.sqrt((3.0 ** 2) * sigma[0] ** 2 + (4.0 ** 2) * sigma[1] ** 2)
    nominal = 3.0 * 2.0 + 4.0 * 1.0
    assert design["design_upper"][0, 0] == pytest.approx(nominal + manual_margin)
    assert design["design_lower"][0, 0] == pytest.approx(nominal - manual_margin)


def test_actuator_envelope_unit_constant_matches_nominal_hard_and_speed_linear_varies():
    from src.torque_constraints import compute_torque_design_data

    def reg(q, dq, ddq):
        return np.array([[1.0]])

    q = np.array([[0.0, 0.0]])
    dq = np.array([[0.0, 1.0]])
    ddq = np.array([[0.0, 0.0]])
    params = np.array([2.0])
    tau_lim = np.array([[-10.0, 10.0]])

    hard = compute_torque_design_data(q, dq, ddq, reg, params, tau_lim, "nominal_hard", {})
    constant = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "actuator_envelope",
        {"envelope_type": "constant"},
    )
    speed = compute_torque_design_data(
        q, dq, ddq, reg, params, tau_lim, "actuator_envelope",
        {"envelope_type": "speed_linear", "speed_linear_slope": 0.5, "velocity_reference": 1.0,
         "min_scale": 0.5, "max_scale": 1.0},
    )

    np.testing.assert_allclose(hard["limit_upper"], constant["limit_upper"], atol=1e-12)
    np.testing.assert_allclose(hard["limit_lower"], constant["limit_lower"], atol=1e-12)
    assert speed["limit_upper"][0, 1] == pytest.approx(5.0)
    assert speed["limit_lower"][0, 1] == pytest.approx(-5.0)


def test_torque_pipeline_end_to_end_nominal_hard(tmp_path):
    cfg = _write_torque_config(tmp_path, torque_method="nominal_hard")
    summary, torque, _ = _run_pipeline(cfg)
    assert summary["torque_nominal_pass"] is True
    assert summary["torque_identified_pass"] is True
    assert float(np.max(torque["identified_ratio"])) <= 1.0 + 1e-6


def test_torque_pipeline_end_to_end_robust_box(tmp_path):
    cfg = _write_torque_config(
        tmp_path,
        torque_method="robust_box",
        torque_constraint={"relative_uncertainty": 0.1, "absolute_uncertainty_floor": 1e-3},
    )
    summary, torque, _ = _run_pipeline(cfg)
    assert summary["torque_identified_pass"] is True
    assert bool(np.asarray(torque["design_pass"]).item()) is True


def test_torque_pipeline_end_to_end_chance(tmp_path):
    cfg = _write_torque_config(
        tmp_path,
        torque_method="chance",
        torque_constraint={
            "relative_stddev": 0.05,
            "absolute_stddev_floor": 1e-3,
            "chance_confidence": 0.95,
        },
    )
    summary, torque, _ = _run_pipeline(cfg)
    assert summary["torque_identified_pass"] is True
    assert bool(np.asarray(torque["design_pass"]).item()) is True
    assert float(np.asarray(torque["chance_quantile"]).item()) == pytest.approx(
        NormalDist().inv_cdf(0.95)
    )


def test_torque_pipeline_end_to_end_actuator_envelope_with_rms(tmp_path):
    cfg = _write_torque_config(
        tmp_path,
        torque_method="actuator_envelope",
        torque_limits=[[-50.0, 50.0], [-50.0, 50.0]],
        torque_constraint={
            "envelope_type": "speed_linear",
            "speed_linear_slope": 0.2,
            "velocity_reference": 2.0,
            "min_scale": 0.6,
            "max_scale": 1.0,
            "rms_limit_ratio": 0.5,
        },
    )
    summary, torque, _ = _run_pipeline(cfg)
    assert summary["torque_identified_pass"] is True
    assert float(np.max(torque["identified_ratio"])) <= 1.0 + 1e-6


def test_torque_pipeline_end_to_end_sequential_redesign_improves_or_matches_initial_ratio(tmp_path):
    cfg = _write_torque_config(
        tmp_path,
        torque_method="sequential_redesign",
        torque_limits=[[-30.0, 30.0], [-30.0, 30.0]],
        torque_constraint={"max_iterations": 2, "convergence_tol": 0.0},
        max_iter=10,
    )
    summary, _, results = _run_pipeline(cfg)
    history = summary["sequential_history"]
    assert len(history) >= 1
    if len(history) > 1:
        assert history[-1]["max_identified_torque_ratio"] <= history[0]["max_identified_torque_ratio"] + 1e-8
    stored_history = results["sequential_history"]
    assert len(stored_history) == len(history)


def test_soft_penalty_end_to_end_higher_weight_reduces_identified_ratio(tmp_path):
    low_cfg = _write_torque_config(
        tmp_path / "low",
        torque_method="soft_penalty",
        torque_limits=[[-10.0, 10.0], [-10.0, 10.0]],
        torque_constraint={"soft_penalty_weight": 1.0, "soft_penalty_smoothing": 0.05},
        max_iter=20,
    )
    high_cfg = _write_torque_config(
        tmp_path / "high",
        torque_method="soft_penalty",
        torque_limits=[[-10.0, 10.0], [-10.0, 10.0]],
        torque_constraint={"soft_penalty_weight": 500.0, "soft_penalty_smoothing": 0.05},
        max_iter=20,
    )
    _, _, _ = _run_pipeline(low_cfg)
    high_summary, _, _ = _run_pipeline(high_cfg)
    low_summary = json.loads((Path(tmp_path / "low" / "out" / "results_summary.json")).read_text(encoding="utf-8"))
    assert high_summary["max_identified_torque_ratio"] <= low_summary["max_identified_torque_ratio"] + 1e-6


@pytest.mark.parametrize(
    "torque_method,torque_constraint",
    [
        ("nominal_hard", {}),
        ("soft_penalty", {"soft_penalty_weight": 100.0, "soft_penalty_smoothing": 0.05}),
        ("robust_box", {"relative_uncertainty": 0.1, "absolute_uncertainty_floor": 1e-3}),
        ("chance", {"relative_stddev": 0.05, "absolute_stddev_floor": 1e-3, "chance_confidence": 0.95}),
        ("actuator_envelope", {"envelope_type": "constant"}),
        ("sequential_redesign", {"max_iterations": 2, "convergence_tol": 0.0}),
    ],
)
def test_stage_7_to_12_shared_torque_harness_runs_one_method_at_a_time(
    tmp_path, torque_method, torque_constraint
):
    cfg = _write_torque_config(
        tmp_path,
        torque_method=torque_method,
        torque_constraint=torque_constraint,
        torque_limits=[[-50.0, 50.0], [-50.0, 50.0]],
        max_iter=10,
    )
    summary, torque, _ = _run_pipeline(cfg)
    assert summary["torque_constraint_method"] == torque_method
    assert summary["torque_limit_source"] == ["json_torque", "json_torque"]
    assert "tau_identified" in torque.files
    assert summary["worst_joint"] in (0, 1)
