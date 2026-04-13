"""Documentation-linked verification tests for the standalone pipeline."""
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_1DOF = str(ASSET_DIR / "SC_1DoF.urdf")
URDF_3DOF = str(ASSET_DIR / "SC_3DoF.urdf")


def _write_config(
    tmp_path,
    urdf_path,
    *,
    n_dof,
    method="newton_euler",
    feasibility="none",
    parameter_bounds=False,
    constraint_style="legacy_excTrajGen",
    filtering=None,
    downsampling=None,
):
    filtering = filtering or {"enabled": False}
    downsampling = downsampling or {"frequency_hz": 0}

    cfg = {
        "urdf_path": urdf_path,
        "output_dir": str(tmp_path / "out"),
        "method": method,
        "joint_limits": {
            "position": [[-1.0, 1.0]] * n_dof,
            "velocity": [[-2.0, 2.0]] * n_dof,
            "acceleration": [[-5.0, 5.0]] * n_dof,
        },
        "excitation": {
            "basis_functions": "cosine",
            "optimize_phase": False,
            "num_harmonics": 1,
            "base_frequency_hz": 0.2,
            "optimize_condition_number": False,
            "constraint_style": constraint_style,
            "optimizer_max_iter": 1,
            "optimizer_pop_size": 2,
            "trajectory_duration_periods": 1,
        },
        "friction": {"model": "none"},
        "identification": {
            "solver": "ols",
            "parameter_bounds": parameter_bounds,
            "feasibility_method": feasibility,
            "data_file": None,
        },
        "filtering": filtering,
        "downsampling": downsampling,
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def test_stage_1_and_3_parser_and_kinematics_build_standalone_model():
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    robot = parse_urdf(URDF_3DOF)
    kin = RobotKinematics(robot)

    assert robot.nDoF == 3
    assert robot.revolute_joint_names == ["Joint_1", "Joint_2", "Joint_3"]
    assert kin.PI.shape == (30, 1)
    np.testing.assert_allclose(
        kin.PI[:10].flatten(),
        np.array([5.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.05]),
        atol=1e-12,
    )


def test_stage_2_joint_limit_extraction_rejects_missing_json_overrides():
    from src.urdf_parser import extract_joint_limits, parse_urdf

    robot = parse_urdf(URDF_1DOF)
    cfg_limits = {
        "position": None,
        "velocity": None,
        "acceleration": None,
    }

    with pytest.raises(ValueError, match="Position limits missing"):
        extract_joint_limits(robot, cfg_limits, logging.getLogger("test"))


def test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques(tmp_path):
    from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_1DOF))
    reg_el, kept = euler_lagrange_regressor_builder(kin, str(tmp_path / "el_cache"))
    pi = kin.PI.flatten()
    rng = np.random.default_rng(7)

    for _ in range(3):
        q = rng.uniform(-0.5, 0.5, kin.nDoF)
        dq = rng.uniform(-1.0, 1.0, kin.nDoF)
        ddq = rng.uniform(-2.0, 2.0, kin.nDoF)
        tau_ne = newton_euler_regressor(kin, q, dq, ddq) @ pi
        tau_el = reg_el(q, dq, ddq) @ pi[kept]
        np.testing.assert_allclose(tau_ne, tau_el, atol=1e-10)


def test_stage_6_friction_regressor_augmentation_matches_supported_models():
    from src.friction import build_friction_regressor

    dq = np.array([-0.5, 0.0, 0.75])
    a = 10.0
    b = 1000.0

    np.testing.assert_equal(build_friction_regressor(dq, "none").shape, (3, 0))
    np.testing.assert_allclose(
        build_friction_regressor(dq, "viscous"),
        np.diag(dq),
        atol=1e-12,
    )

    with np.errstate(over="ignore"):
        expected_cp = np.diag(1.0 / (1.0 + np.exp(a - b * dq)))
        expected_cn = np.diag(-1.0 / (1.0 + np.exp(a + b * dq)))

    with np.errstate(over="ignore"):
        np.testing.assert_allclose(
            build_friction_regressor(dq, "coulomb"),
            np.hstack((expected_cp, expected_cn)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            build_friction_regressor(dq, "viscous_coulomb"),
            np.hstack((np.diag(dq), expected_cp, expected_cn)),
            atol=1e-12,
        )


def test_stage_6_sine_basis_enforces_boundary_conditions_on_integer_periods():
    from src.trajectory import build_frequencies, fourier_trajectory

    freqs = build_frequencies(0.2, 3)
    q0 = np.array([0.1, -0.2])
    params = np.array([0.12, -0.08, 0.04, -0.03, 0.07, -0.02])
    T = 2 / 0.2
    t = np.linspace(0.0, T, 2001)

    q, dq, ddq = fourier_trajectory(params, freqs, t, q0, "sine", False)

    np.testing.assert_allclose(q[:, 0], q0, atol=1e-14)
    np.testing.assert_allclose(dq[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(dq[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(ddq[:, -1], 0.0, atol=1e-12)


def test_stage_6_noninteger_sine_periods_are_rejected_by_config(tmp_path):
    from src.config_loader import load_config

    cfg_path = tmp_path / "bad_sine.json"
    cfg_path.write_text(
        json.dumps(
            {
                "urdf_path": URDF_1DOF,
                "output_dir": str(tmp_path / "out"),
                "excitation": {
                    "basis_functions": "sine",
                    "trajectory_duration_periods": 1.5,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer"):
        load_config(str(cfg_path))


def test_stage_6_excitation_styles_dispatch_to_distinct_solver_paths(monkeypatch):
    from src.excitation import optimise_excitation

    calls = {"de": 0, "minimize": 0}

    def fake_de(cost, bounds, max_iter, pop_size):
        calls["de"] += 1
        x = np.zeros(len(bounds))
        return SimpleNamespace(x=x, fun=float(cost(x)))

    def fake_minimize(fun, x0, method, constraints, bounds, options):
        calls["minimize"] += 1
        return SimpleNamespace(
            x=np.zeros_like(x0),
            fun=float(fun(np.zeros_like(x0))),
            success=True,
            nit=1,
            message="ok",
        )

    monkeypatch.setattr("src.excitation._run_differential_evolution", fake_de)
    monkeypatch.setattr("src.excitation.minimize", fake_minimize)

    class DummyKin:
        nDoF = 1

    q_lim = np.array([[-1.0, 1.0]])
    dq_lim = np.array([[-2.0, 2.0]])
    ddq_lim = np.array([[-5.0, 5.0]])
    base_cfg = {
        "basis_functions": "cosine",
        "optimize_phase": False,
        "num_harmonics": 1,
        "base_frequency_hz": 0.2,
        "optimize_condition_number": False,
        "optimizer_max_iter": 1,
        "optimizer_pop_size": 2,
        "trajectory_duration_periods": 1,
    }

    for style in ("legacy_excTrajGen", "urdf_reference", "literature_standard"):
        cfg = dict(base_cfg)
        cfg["constraint_style"] = style
        optimise_excitation(DummyKin(), cfg, q_lim, dq_lim, ddq_lim)

    assert calls["de"] == 2
    assert calls["minimize"] == 1


def test_stage_8_observation_matrix_matches_manual_stacking_equation():
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.observation_matrix import build_observation_matrix
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_3DOF))
    rng = np.random.default_rng(21)
    q = rng.uniform(-0.5, 0.5, size=(12, 3))
    dq = rng.uniform(-1.0, 1.0, size=(12, 3))
    ddq = rng.uniform(-2.0, 2.0, size=(12, 3))
    tau = rng.uniform(-0.5, 0.5, size=(12, 3))
    cfg = {
        "friction": {"model": "none"},
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }

    W, tau_vec = build_observation_matrix(
        q,
        dq,
        ddq,
        tau,
        lambda qv, dqv, ddqv: newton_euler_regressor(kin, qv, dqv, ddqv),
        cfg,
        100.0,
    )
    W_manual = np.vstack(
        [newton_euler_regressor(kin, q[k], dq[k], ddq[k]) for k in range(q.shape[0])]
    )

    np.testing.assert_allclose(W, W_manual, atol=1e-12)
    np.testing.assert_allclose(tau_vec, tau.reshape(-1), atol=1e-12)


def test_stage_8_filtering_happens_before_downsampling():
    from src.filtering import apply_filter
    from src.observation_matrix import build_observation_matrix

    fs = 1000.0
    t = np.arange(1000) / fs
    low = np.sin(2 * np.pi * 5 * t)
    high = 0.5 * np.sin(2 * np.pi * 200 * t)
    q = (low + high).reshape(-1, 1)
    zeros = np.zeros_like(q)
    cfg = {
        "friction": {"model": "none"},
        "filtering": {
            "enabled": True,
            "cutoff_frequency_hz": 50.0,
            "filter_order": 4,
        },
        "downsampling": {"frequency_hz": 100.0},
    }

    W, _ = build_observation_matrix(
        q,
        zeros,
        zeros,
        zeros,
        lambda qv, dqv, ddqv: np.array([[qv[0]]]),
        cfg,
        fs,
    )
    filtered_then_ds = apply_filter(q, fs, cfg["filtering"])[::10, 0]
    raw_ds = q[::10, 0]

    np.testing.assert_allclose(W[:, 0], filtered_then_ds, atol=1e-12)
    assert np.max(np.abs(W[:, 0] - raw_ds)) > 1e-3


@pytest.mark.parametrize("method", ["newton_euler", "euler_lagrange"])
def test_stage_9_base_parameter_reduction_preserves_observation_equation_for_ne_and_el(
    tmp_path, method
):
    from src.base_parameters import compute_base_parameters
    from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_1DOF))
    rng = np.random.default_rng(11)

    if method == "newton_euler":
        regressor_fn = lambda qv, dqv, ddqv: newton_euler_regressor(kin, qv, dqv, ddqv)
        pi = kin.PI.flatten()
    else:
        regressor_fn, kept = euler_lagrange_regressor_builder(kin, str(tmp_path / "el_cache"))
        pi = kin.PI.flatten()[kept]

    rows = []
    for _ in range(8):
        q = rng.uniform(-0.5, 0.5, kin.nDoF)
        dq = rng.uniform(-1.0, 1.0, kin.nDoF)
        ddq = rng.uniform(-2.0, 2.0, kin.nDoF)
        rows.append(regressor_fn(q, dq, ddq))
    W = np.vstack(rows)

    W_base, P, kept_cols, rank, pi_base = compute_base_parameters(W, pi)
    np.testing.assert_allclose(W @ pi, W_base @ pi_base, atol=1e-10)
    assert rank == W_base.shape[1]
    assert len(kept_cols) == rank
    assert P.shape[0] == rank


def test_stage_10_parameter_bounds_enable_bounded_ls(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_1DOF,
        n_dof=1,
        parameter_bounds=True,
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    summary = json.loads((tmp_path / "out" / "results_summary.json").read_text(encoding="utf-8"))

    assert summary["solver"] == "bounded_ls"


def test_stage_11_pseudo_inertia_checks_report_standard_rigid_body_failures():
    from src.feasibility import check_feasibility

    pi_bad = np.array([-1.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.1])
    report, feasible, _ = check_feasibility(pi_bad, 1)
    issues = " | ".join(report[0]["issues"])

    assert not feasible
    assert "Non-positive mass" in issues
    assert "Inertia not PSD" in issues
    assert "Triangle ineq." in issues
    assert "Pseudo-inertia NOT PSD" in issues


def test_stage_11_euler_lagrange_rejects_constrained_feasibility_modes(tmp_path):
    from src.config_loader import load_config

    cfg_path = tmp_path / "bad_el_lmi.json"
    cfg_path.write_text(
        json.dumps(
            {
                "urdf_path": URDF_1DOF,
                "output_dir": str(tmp_path / "out"),
                "method": "euler_lagrange",
                "identification": {"feasibility_method": "lmi"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="euler_lagrange"):
        load_config(str(cfg_path))


def test_stage_12_pipeline_success_and_feasibility_are_distinct_for_unconstrained_run(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_3DOF,
        n_dof=3,
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    results = np.load(tmp_path / "out" / "identification_results.npz", allow_pickle=True)
    log_text = (tmp_path / "out" / "pipeline.log").read_text(encoding="utf-8")

    assert not bool(results["feasible"])
    assert "PIPELINE COMPLETED SUCCESSFULLY" in log_text


def test_stage_12_constrained_lmi_returns_feasible_newton_euler_model(tmp_path):
    from src.feasibility import is_pseudo_inertia_psd
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_1DOF,
        n_dof=1,
        feasibility="lmi",
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    results = np.load(tmp_path / "out" / "identification_results.npz", allow_pickle=True)

    assert bool(results["feasible"])
    assert bool(results["solved_in_full_space"])
    assert is_pseudo_inertia_psd(results["pi_identified"][:10])
