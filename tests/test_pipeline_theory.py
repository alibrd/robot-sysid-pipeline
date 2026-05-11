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
URDF_RRBOT = str(ASSET_DIR / "RRBot_single.urdf")
URDF_PENDULUM = str(ASSET_DIR / "DrakePendulum_1DoF.urdf")
URDF_FINGEREDU = str(ASSET_DIR / "FingerEdu_3DoF.xacro")
URDF_DEFAULT = URDF_RRBOT


def _write_config(
    tmp_path,
    urdf_path,
    *,
    n_dof,
    method="newton_euler",
    feasibility="none",
    parameter_bounds=False,
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
            "optimizer_max_iter": 1,
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

    robot = parse_urdf(URDF_DEFAULT)
    kin = RobotKinematics(robot)

    print("\nSTAGE 1 & 3: URDF parsing and inertial-parameter vector construction")
    print(f"  Parsed nDoF          = {robot.nDoF}")
    print(f"  Revolute joint names = {robot.revolute_joint_names}")
    print(f"  PI vector shape      = {kin.PI.shape}")
    expected = np.array([1.0, 0.0, 0.0, 0.45, 1.2025, 0.0, 0.0, 1.2025, 0.0, 1.0])
    actual = kin.PI[:10].flatten()
    print(f"  PI link-1 expected   = {expected}")
    print(f"  PI link-1 actual     = {actual}")
    max_dev = np.max(np.abs(actual - expected))
    print(f"  max|PI_actual - PI_expected| = {max_dev:.2e}")

    assert robot.nDoF == 2
    assert robot.revolute_joint_names == ["single_rrbot_joint1", "single_rrbot_joint2"]
    assert kin.PI.shape == (20, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    print(f"  VERIFIED: URDF parsed correctly, PI vector matches expected values (atol=1e-12)")


def test_stage_2_joint_limit_extraction_rejects_missing_json_overrides():
    from src.urdf_parser import extract_joint_limits, parse_urdf

    robot = parse_urdf(URDF_DEFAULT)
    cfg_limits = {
        "position": None,
        "velocity": None,
        "acceleration": None,
    }

    print("\nSTAGE 2: Missing joint limits must be rejected early")
    with pytest.raises(ValueError, match="Position limits missing"):
        extract_joint_limits(robot, cfg_limits, logging.getLogger("test"))
    print("  ValueError raised with 'Position limits missing'")
    print("  VERIFIED: Pipeline fails early when required joint limits are absent")


def test_stage_4_and_5_newton_euler_and_euler_lagrange_match_shared_torques(tmp_path):
    from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
    reg_el, kept = euler_lagrange_regressor_builder(kin, str(tmp_path / "el_cache"))
    pi = kin.PI.flatten()
    rng = np.random.default_rng(7)

    print(f"\nSTAGE 4 & 5: NE and EL regressors must produce identical torques (2-DoF RRBot)")
    for i in range(3):
        q = rng.uniform(-0.5, 0.5, kin.nDoF)
        dq = rng.uniform(-1.0, 1.0, kin.nDoF)
        ddq = rng.uniform(-2.0, 2.0, kin.nDoF)
        tau_ne = newton_euler_regressor(kin, q, dq, ddq) @ pi
        tau_el = reg_el(q, dq, ddq) @ pi[kept]
        err = np.max(np.abs(tau_ne - tau_el))
        print(f"  State {i+1}: tau_ne={tau_ne}, tau_el={tau_el}, max|diff|={err:.2e}")
        np.testing.assert_allclose(tau_ne, tau_el, atol=1e-10)
    print("  VERIFIED: NE and EL torques agree across 3 random states (atol=1e-10)")


def test_stage_6_friction_regressor_augmentation_matches_supported_models():
    from src.friction import build_friction_regressor

    dq = np.array([-0.5, 0.0, 0.75])
    a = 10.0
    b = 1000.0

    print("\nSTAGE 7 (friction): Friction regressor blocks match analytical definitions")

    Yf_none = build_friction_regressor(dq, "none")
    print(f"  'none'    : shape={Yf_none.shape}")
    np.testing.assert_equal(Yf_none.shape, (3, 0))

    Yf_visc = build_friction_regressor(dq, "viscous")
    err_visc = np.max(np.abs(Yf_visc - np.diag(dq)))
    print(f"  'viscous' : shape={Yf_visc.shape}, max|err|={err_visc:.2e}")
    np.testing.assert_allclose(Yf_visc, np.diag(dq), atol=1e-12)

    with np.errstate(over="ignore"):
        expected_cp = np.diag(1.0 / (1.0 + np.exp(a - b * dq)))
        expected_cn = np.diag(-1.0 / (1.0 + np.exp(a + b * dq)))

    with np.errstate(over="ignore"):
        Yf_coul = build_friction_regressor(dq, "coulomb")
        err_coul = np.max(np.abs(Yf_coul - np.hstack((expected_cp, expected_cn))))
        print(f"  'coulomb' : shape={Yf_coul.shape}, max|err|={err_coul:.2e}")
        np.testing.assert_allclose(Yf_coul, np.hstack((expected_cp, expected_cn)), atol=1e-12)

        Yf_vc = build_friction_regressor(dq, "viscous_coulomb")
        err_vc = np.max(np.abs(Yf_vc - np.hstack((np.diag(dq), expected_cp, expected_cn))))
        print(f"  'viscous_coulomb': shape={Yf_vc.shape}, max|err|={err_vc:.2e}")
        np.testing.assert_allclose(Yf_vc, np.hstack((np.diag(dq), expected_cp, expected_cn)), atol=1e-12)

    print("  VERIFIED: All friction models match analytical definitions (atol=1e-12)")


def test_stage_6_sine_basis_enforces_boundary_conditions_on_integer_periods():
    from src.trajectory import build_frequencies, fourier_trajectory

    freqs = build_frequencies(0.2, 3)
    q0 = np.array([0.1, -0.2])
    params = np.array([0.12, -0.08, 0.04, -0.03, 0.07, -0.02])
    T = 2 / 0.2
    t = np.linspace(0.0, T, 2001)

    q, dq, ddq = fourier_trajectory(params, freqs, t, q0, "sine", False)

    print("\nSTAGE 6: Sine basis boundary conditions on integer periods")
    print(f"  q(0)    = {q[:, 0]},   expected q0 = {q0}")
    print(f"  dq(0)   = {dq[:, 0]},  expected = 0")
    print(f"  dq(T)   = {dq[:, -1]}, expected = 0")
    print(f"  ddq(T)  = {ddq[:, -1]}, expected = 0")
    print(f"  |q(0)-q0|  = {np.max(np.abs(q[:, 0] - q0)):.2e}")
    print(f"  |dq(0)|    = {np.max(np.abs(dq[:, 0])):.2e}")
    print(f"  |dq(T)|    = {np.max(np.abs(dq[:, -1])):.2e}")
    print(f"  |ddq(T)|   = {np.max(np.abs(ddq[:, -1])):.2e}")

    np.testing.assert_allclose(q[:, 0], q0, atol=1e-14)
    np.testing.assert_allclose(dq[:, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(dq[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(ddq[:, -1], 0.0, atol=1e-12)
    print("  VERIFIED: q(0)=q0, dq(0)=0, dq(T)=0, ddq(T)=0 on integer periods")


def test_stage_6_noninteger_sine_periods_are_rejected_by_config(tmp_path):
    from src.config_loader import load_config

    cfg_path = tmp_path / "bad_sine.json"
    cfg_path.write_text(
        json.dumps(
            {
                "urdf_path": URDF_DEFAULT,
                "output_dir": str(tmp_path / "out"),
                "excitation": {
                    "basis_functions": "sine",
                    "trajectory_duration_periods": 1.5,
                },
            }
        ),
        encoding="utf-8",
    )

    print("\nSTAGE 6: Noninteger sine periods must be rejected by config validation")
    with pytest.raises(ValueError, match="integer"):
        load_config(str(cfg_path))
    print("  ValueError raised matching 'integer'")
    print("  VERIFIED: Config rejects sine basis with trajectory_duration_periods=1.5")


def test_stage_6_excitation_uses_single_slsqp_path(monkeypatch):
    from src.excitation import optimise_excitation

    calls = {"minimize": 0}

    def fake_minimize(fun, x0, method, constraints, bounds, options):
        calls["minimize"] += 1
        return SimpleNamespace(
            x=np.zeros_like(x0),
            fun=float(fun(np.zeros_like(x0))),
            success=True,
            nit=1,
            message="ok",
        )

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
        "trajectory_duration_periods": 1,
    }

    print("\nSTAGE 6: Excitation uses the single literature-standard SLSQP path")
    optimise_excitation(DummyKin(), dict(base_cfg), q_lim, dq_lim, ddq_lim)

    print(f"  scipy.minimize calls = {calls['minimize']} (expected 1)")
    assert calls["minimize"] == 1
    print("  VERIFIED: Excitation dispatches to the single SLSQP solver path")


def test_stage_6_initial_guess_keeps_high_harmonic_ddq_margin(monkeypatch):
    from src.excitation import optimise_excitation
    from src.torque_constraints import validation_time_vector
    from src.trajectory import fourier_trajectory

    captured = {}

    def fake_minimize(fun, x0, method, constraints, bounds, options):
        captured["x0"] = x0.copy()
        return SimpleNamespace(
            x=x0.copy(),
            fun=float(fun(x0)),
            success=True,
            nit=1,
            message="ok",
        )

    monkeypatch.setattr("src.excitation.minimize", fake_minimize)

    class DummyKin:
        nDoF = 2

    q_lim = np.array([[-1.57, 1.57], [-1.57, 1.57]])
    dq_lim = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    ddq_lim = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    cfg = {
        "basis_functions": "both",
        "optimize_phase": False,
        "num_harmonics": 20,
        "base_frequency_hz": 0.5,
        "optimize_condition_number": False,
        "optimizer_max_iter": 1,
        "trajectory_duration_periods": 60,
        "torque_constraint_method": "none",
        "torque_validation_oversample_factor": 5,
    }

    result = optimise_excitation(DummyKin(), cfg, q_lim, dq_lim, ddq_lim)
    t_dense = validation_time_vector(
        result["freqs"],
        cfg["base_frequency_hz"],
        cfg["trajectory_duration_periods"],
        cfg["torque_validation_oversample_factor"],
    )
    _, _, ddq = fourier_trajectory(
        captured["x0"],
        result["freqs"],
        t_dense,
        result["q0"],
        cfg["basis_functions"],
        cfg["optimize_phase"],
    )

    assert np.max(np.abs(ddq)) < 2.7


def test_stage_8_observation_matrix_matches_manual_stacking_equation():
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.observation_matrix import build_observation_matrix
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
    rng = np.random.default_rng(21)
    q = rng.uniform(-0.5, 0.5, size=(12, kin.nDoF))
    dq = rng.uniform(-1.0, 1.0, size=(12, kin.nDoF))
    ddq = rng.uniform(-2.0, 2.0, size=(12, kin.nDoF))
    tau = rng.uniform(-0.5, 0.5, size=(12, kin.nDoF))
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

    print("\nSTAGE 8: Observation matrix W must equal manual row-by-row stacking")
    print(f"  W shape       = {W.shape}")
    print(f"  W_manual shape= {W_manual.shape}")
    w_err = np.max(np.abs(W - W_manual))
    t_err = np.max(np.abs(tau_vec - tau.reshape(-1)))
    print(f"  max|W - W_manual|       = {w_err:.2e}")
    print(f"  max|tau_vec - tau_flat|  = {t_err:.2e}")

    np.testing.assert_allclose(W, W_manual, atol=1e-12)
    np.testing.assert_allclose(tau_vec, tau.reshape(-1), atol=1e-12)
    print("  VERIFIED: W matches manual stacking and tau_vec matches flat tau (atol=1e-12)")


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

    print("\nSTAGE 8: Filtering must happen before downsampling")
    filter_match = np.max(np.abs(W[:, 0] - filtered_then_ds))
    raw_deviation = np.max(np.abs(W[:, 0] - raw_ds))
    print(f"  max|W - filtered_then_ds| = {filter_match:.2e} (should be ~0)")
    print(f"  max|W - raw_ds|           = {raw_deviation:.2e} (should be >> 0)")

    np.testing.assert_allclose(W[:, 0], filtered_then_ds, atol=1e-12)
    assert raw_deviation > 1e-3
    print("  VERIFIED: Pipeline filters before downsampling (not the reverse)")


@pytest.mark.parametrize("method", ["newton_euler", "euler_lagrange"])
def test_stage_9_base_parameter_reduction_preserves_observation_equation_for_ne_and_el(
    tmp_path, method
):
    from src.base_parameters import compute_base_parameters
    from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
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

    print(f"\nSTAGE 9: Base-parameter reduction preserves observation equation ({method})")
    print(f"  Full W shape  = {W.shape} ({W.shape[1]} params)")
    print(f"  Base W shape  = {W_base.shape} (rank={rank})")
    obs_err = np.max(np.abs(W @ pi - W_base @ pi_base))
    print(f"  max|W*pi - W_b*pi_b| = {obs_err:.2e}")

    np.testing.assert_allclose(W @ pi, W_base @ pi_base, atol=1e-10)
    assert rank == W_base.shape[1]
    assert len(kept_cols) == rank
    assert P.shape[0] == rank
    print(f"  VERIFIED: W*pi == W_b*pi_b for {method} (atol=1e-10, rank={rank})")


def test_stage_10_parameter_bounds_enable_bounded_ls(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_DEFAULT,
        n_dof=2,
        parameter_bounds=True,
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    summary = json.loads((tmp_path / "out" / "results_summary.json").read_text(encoding="utf-8"))

    print("\nSTAGE 10: parameter_bounds=true must switch solver to bounded_ls")
    print(f"  Solver in results_summary.json = '{summary['solver']}'")
    assert summary["solver"] == "bounded_ls"
    print("  VERIFIED: Pipeline auto-switched from OLS to bounded_ls when parameter_bounds=true")


def test_stage_10_viscous_clamp_uses_friction_tail_for_reduced_vectors():
    from src.pipeline import _clamp_negative_viscous_damping

    pi = np.array([
        10.0, 11.0, 12.0, 13.0, 14.0,  # reduced rigid-body block
        -0.3, 0.2,                      # Fv tail for two joints
        0.7, -0.8, 0.9, -1.1,           # Coulomb terms remain unconstrained
    ])

    out = _clamp_negative_viscous_damping(
        pi, n_dof=2, friction_model="viscous_coulomb",
        log=logging.getLogger("test"),
    )

    expected = np.array([
        10.0, 11.0, 12.0, 13.0, 14.0,
        0.0, 0.2,
        0.7, -0.8, 0.9, -1.1,
    ])
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(
        pi,
        [10.0, 11.0, 12.0, 13.0, 14.0, -0.3, 0.2, 0.7, -0.8, 0.9, -1.1],
    )


def test_stage_11_pseudo_inertia_checks_report_standard_rigid_body_failures():
    from src.feasibility import check_feasibility

    pi_bad = np.array([-1.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.1])
    report, feasible, _ = check_feasibility(pi_bad, 1)
    issues = " | ".join(report[0]["issues"])

    print("\nSTAGE 11: Pseudo-inertia checks must detect physically invalid bodies")
    print(f"  pi_bad      = {pi_bad}")
    print(f"  feasible    = {feasible}")
    print(f"  Issues: {issues}")

    assert not feasible
    assert "Non-positive mass" in issues
    assert "Inertia not PSD" in issues
    assert "Triangle ineq." in issues
    assert "Pseudo-inertia NOT PSD" in issues
    print("  VERIFIED: All standard rigid-body failure conditions detected")


def test_stage_11_euler_lagrange_rejects_constrained_feasibility_modes(tmp_path):
    from src.config_loader import load_config

    cfg_path = tmp_path / "bad_el_lmi.json"
    cfg_path.write_text(
        json.dumps(
            {
                "urdf_path": URDF_DEFAULT,
                "output_dir": str(tmp_path / "out"),
                "method": "euler_lagrange",
                "identification": {"feasibility_method": "lmi"},
            }
        ),
        encoding="utf-8",
    )

    print("\nSTAGE 11: EL method must reject constrained feasibility modes (lmi/cholesky)")
    with pytest.raises(ValueError, match="euler_lagrange"):
        load_config(str(cfg_path))
    print("  ValueError raised matching 'euler_lagrange'")
    print("  VERIFIED: euler_lagrange + feasibility_method='lmi' is rejected at config load")


def test_stage_12_pipeline_success_and_feasibility_are_distinct_for_unconstrained_run(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_DEFAULT,
        n_dof=2,
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    results = np.load(tmp_path / "out" / "identification_results.npz", allow_pickle=True)
    log_text = (tmp_path / "out" / "pipeline.log").read_text(encoding="utf-8")

    print("\nSTAGE 12: Pipeline success and physical feasibility are distinct")
    print(f"  feasible flag         = {bool(results['feasible'])}")
    completed = "PIPELINE COMPLETED SUCCESSFULLY" in log_text
    print(f"  pipeline completed    = {completed}")

    assert not bool(results["feasible"])
    assert completed
    print("  VERIFIED: Pipeline completed successfully but feasible=False (unconstrained run)")


def test_stage_12_constrained_lmi_returns_feasible_newton_euler_model(tmp_path):
    from src.feasibility import is_pseudo_inertia_psd
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_DEFAULT,
        n_dof=2,
        feasibility="lmi",
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    results = np.load(tmp_path / "out" / "identification_results.npz", allow_pickle=True)

    print("\nSTAGE 12: LMI-constrained NE identification must return a feasible model")
    print(f"  feasible            = {bool(results['feasible'])}")
    print(f"  solved_in_full_space= {bool(results['solved_in_full_space'])}")
    pi_id = results["pi_identified"]
    for link in range(2):
        from src.feasibility import pseudo_inertia_matrix
        J = pseudo_inertia_matrix(pi_id[link * 10:(link + 1) * 10])
        eigs = np.linalg.eigvalsh(J)
        print(f"  Link {link+1} pseudo-inertia eigenvalues = {eigs}")

    assert bool(results["feasible"])
    assert bool(results["solved_in_full_space"])
    assert is_pseudo_inertia_psd(results["pi_identified"][:10])
    print("  VERIFIED: LMI-constrained model is feasible with J_i >= 0 for all links")


def test_stage_1_reference_models_remain_supported():
    from src.urdf_parser import parse_urdf

    r1 = parse_urdf(URDF_PENDULUM)
    r3 = parse_urdf(URDF_FINGEREDU)

    print("\nSTAGE 1: Additional URDF fixtures remain supported")
    print(f"  DrakePendulum_1DoF nDoF = {r1.nDoF}")
    print(f"  FingerEdu_3DoF nDoF = {r3.nDoF}")
    assert r1.nDoF == 1
    assert r3.nDoF == 3
    print("  VERIFIED: DrakePendulum_1DoF and FingerEdu_3DoF fixtures parse correctly")


# ---------------------------------------------------------------------------
# Phase 2 new tests
# ---------------------------------------------------------------------------


def test_stage_7_synthetic_tau_equals_regressor_times_pi(tmp_path):
    """Run the pipeline's synthetic-data path, then independently reconstruct
    tau = Y @ pi from the saved excitation trajectory and verify consistency."""
    from src.base_parameters import compute_base_parameters
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.observation_matrix import build_observation_matrix
    from src.pipeline import SystemIdentificationPipeline
    from src.trajectory import fourier_trajectory
    from src.urdf_parser import parse_urdf

    # 1. Run the full pipeline (synthetic data, no filtering/downsampling)
    cfg_path = _write_config(tmp_path, URDF_DEFAULT, n_dof=2)
    SystemIdentificationPipeline(str(cfg_path)).run()

    # 2. Load saved artifacts
    exc = np.load(tmp_path / "out" / "excitation_trajectory.npz", allow_pickle=True)
    results = np.load(tmp_path / "out" / "identification_results.npz", allow_pickle=True)
    pi_base_pipeline = results["pi_base"]

    # 3. Independently reconstruct q/dq/ddq from the saved excitation params
    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
    pi = kin.PI.flatten()
    cfg_json = json.loads(cfg_path.read_text(encoding="utf-8"))
    f0 = cfg_json["excitation"]["base_frequency_hz"]
    tf = cfg_json["excitation"].get("trajectory_duration_periods", 1) / f0
    freqs = exc["freqs"]
    data_fs = 2.0 * freqs[-1] * 10  # same formula as pipeline.py
    t_data = np.arange(0, tf, 1.0 / data_fs)
    q_t, dq_t, ddq_t = fourier_trajectory(
        exc["params"], freqs, t_data, exc["q0"],
        str(exc["basis"]), bool(exc["optimize_phase"]),
    )
    N = t_data.size

    # 4. Independently compute tau = Y @ pi for each sample
    tau_indep = np.zeros((N, kin.nDoF))
    for k in range(N):
        Y_k = newton_euler_regressor(kin, q_t[:, k], dq_t[:, k], ddq_t[:, k])
        tau_indep[k] = Y_k @ pi

    # 5. Build observation matrix from independently reconstructed data
    obs_cfg = {
        "friction": {"model": "none"},
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }
    reg_fn = lambda qv, dqv, ddqv: newton_euler_regressor(kin, qv, dqv, ddqv)
    W, tau_vec = build_observation_matrix(
        q_t.T, dq_t.T, ddq_t.T, tau_indep, reg_fn, obs_cfg, data_fs,
    )

    # 6. Key check 1: observation equation tau_vec == W @ pi
    obs_err = np.max(np.abs(tau_vec - W @ pi))

    # 7. Key check 2: pipeline's identified base params match true projection
    _, _, _, _, pi_b_true = compute_base_parameters(W, pi)
    param_err = np.max(np.abs(pi_base_pipeline - pi_b_true))

    print(f"\nSTAGE 7: Synthetic data must satisfy tau_k = Y_k @ pi")
    print(f"  nDoF = {kin.nDoF}, n_samples = {N}, PI length = {len(pi)}")
    print(f"  max|tau_vec - W @ pi|           = {obs_err:.2e}")
    print(f"  max|pi_base_pipeline - pi_true| = {param_err:.2e}")

    np.testing.assert_allclose(tau_vec, W @ pi, atol=1e-12)
    np.testing.assert_allclose(pi_base_pipeline, pi_b_true, atol=1e-10)
    print("  VERIFIED: Pipeline synthetic-data path produces tau = Y @ pi and recovers true base params")


def test_stage_10_ols_recovers_exact_base_parameters_from_noiseless_data():
    """OLS must recover exact base parameters when W has full column rank."""
    from src.solver import solve_identification

    rng = np.random.default_rng(99)
    n_params = 7
    n_equations = 50
    W_base = rng.standard_normal((n_equations, n_params))
    pi_base_true = rng.uniform(-2.0, 2.0, n_params)
    tau = W_base @ pi_base_true

    pi_hat, residual, info = solve_identification(W_base, tau, solver="ols")

    print("\nSTAGE 10: OLS must recover exact base parameters from noiseless data")
    print(f"  W_base shape    = {W_base.shape}")
    print(f"  pi_base_true    = {pi_base_true}")
    print(f"  pi_hat          = {pi_hat}")
    max_err = np.max(np.abs(pi_hat - pi_base_true))
    print(f"  max|pi_hat - pi_true| = {max_err:.2e}")
    print(f"  residual              = {residual:.2e}")

    np.testing.assert_allclose(pi_hat, pi_base_true, atol=1e-10)
    print("  VERIFIED: OLS recovers exact base parameters (atol=1e-10)")


def test_stage_11_pseudo_inertia_roundtrip():
    """pi -> J -> pi must recover the original parameter vector."""
    from src.feasibility import pi_from_pseudo_inertia_matrix, pseudo_inertia_matrix

    pi = np.array([2.5, 0.1, -0.2, 0.05, 0.3, 0.01, -0.02, 0.25, 0.005, 0.35])
    J = pseudo_inertia_matrix(pi)
    pi_back = pi_from_pseudo_inertia_matrix(J)

    print("\nSTAGE 11: Pseudo-inertia roundtrip pi -> J -> pi")
    print(f"  pi_original    = {pi}")
    print(f"  pi_recovered   = {pi_back}")
    max_dev = np.max(np.abs(pi_back - pi))
    print(f"  max|pi_back - pi| = {max_dev:.2e}")

    np.testing.assert_allclose(pi_back, pi, atol=1e-14)
    print("  VERIFIED: pi -> J -> pi roundtrip recovers original parameters (atol=1e-14)")


def test_stage_11_cholesky_solver_guarantees_psd():
    """Parameters produced by the Cholesky solver path must have J >= 0."""
    from src.feasibility import is_pseudo_inertia_psd, pseudo_inertia_matrix
    from src.solver import _solve_cholesky

    rng = np.random.default_rng(42)
    nDoF = 1
    W = rng.standard_normal((30, 10))
    tau = rng.standard_normal(30)
    P_mat = np.eye(10)

    pi_hat, res, info = _solve_cholesky(W, tau, nDoF, "ols", None, P_mat)

    print("\nSTAGE 11: Cholesky solver must produce pseudo-inertia PSD output")
    print(f"  solved_in_full_space = {info['solved_in_full_space']}")
    J = pseudo_inertia_matrix(pi_hat[:10])
    eigs = np.linalg.eigvalsh(J)
    print(f"  Pseudo-inertia eigenvalues = {eigs}")
    psd = is_pseudo_inertia_psd(pi_hat[:10])
    print(f"  is_pseudo_inertia_psd      = {psd}")

    assert info["solved_in_full_space"]
    assert psd
    print("  VERIFIED: Cholesky solver guarantees J >= 0 by construction")


def test_stage_12_cholesky_constrained_produces_feasible_model(tmp_path):
    """Cholesky reparameterisation must produce pseudo-inertia PSD result."""
    from src.feasibility import is_pseudo_inertia_psd, pseudo_inertia_matrix
    from src.pipeline import SystemIdentificationPipeline

    cfg_path = _write_config(
        tmp_path,
        URDF_DEFAULT,
        n_dof=2,
        feasibility="cholesky",
    )

    SystemIdentificationPipeline(str(cfg_path)).run()
    results = np.load(tmp_path / "out" / "identification_results.npz", allow_pickle=True)

    nDoF = int(results["nDoF"])
    pi_corrected = results["pi_corrected"]

    print("\nSTAGE 12: Cholesky-constrained pipeline must produce a feasible model")
    print(f"  residual = {float(results['residual']):.6f}")
    print(f"  feasible = {bool(results['feasible'])}")
    print(f"  nDoF = {nDoF}, pi_corrected length = {len(pi_corrected)}")

    all_psd = True
    for link in range(nDoF):
        block = pi_corrected[link * 10 : (link + 1) * 10]
        J = pseudo_inertia_matrix(block)
        eigs = np.linalg.eigvalsh(J)
        psd = is_pseudo_inertia_psd(block)
        print(f"  Link {link + 1} pseudo-inertia eigenvalues = {eigs}, PSD = {psd}")
        all_psd = all_psd and psd

    assert bool(results["feasible"]), "Pipeline feasibility flag must be True"
    assert all_psd, "All link pseudo-inertia matrices must be PSD"
    assert float(results["residual"]) < 1.0
    print("  VERIFIED: Cholesky-constrained pipeline produces feasible pseudo-inertia for all links")
