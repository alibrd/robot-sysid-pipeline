"""Slower literature-verification tests for the standalone pipeline."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_RRBOT = str(ASSET_DIR / "RRBot_single.urdf")
URDF_1DOF = str(ASSET_DIR / "SC_1DoF.urdf")
URDF_3DOF = str(ASSET_DIR / "SC_3DoF.urdf")
URDF_DEFAULT = URDF_RRBOT


@pytest.mark.slow
def test_slow_ne_el_regressors_match_across_random_default_states(tmp_path):
    from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
    reg_el, kept = euler_lagrange_regressor_builder(kin, str(tmp_path / "el_cache"))
    pi = kin.PI.flatten()
    rng = np.random.default_rng(42)

    print(f"\nSTAGE 4 & 5 (slow): NE/EL torque agreement across 10 random states (2-DoF RRBot)")
    for i in range(10):
        q = rng.uniform(-1.0, 1.0, kin.nDoF)
        dq = rng.uniform(-2.0, 2.0, kin.nDoF)
        ddq = rng.uniform(-3.0, 3.0, kin.nDoF)
        tau_ne = newton_euler_regressor(kin, q, dq, ddq) @ pi
        tau_el = reg_el(q, dq, ddq) @ pi[kept]
        err = np.max(np.abs(tau_ne - tau_el))
        print(f"  State {i+1:2d}: max|tau_ne - tau_el| = {err:.2e}")
        np.testing.assert_allclose(tau_ne, tau_el, atol=1e-10)
    print("  VERIFIED: NE and EL torques agree across 10 random states (atol=1e-10)")


@pytest.mark.slow
def test_slow_base_parameter_reduction_preserves_multiple_random_default_observation_matrices():
    from src.base_parameters import compute_base_parameters
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
    pi = kin.PI.flatten()
    rng = np.random.default_rng(123)

    print(f"\nSTAGE 9 (slow): Base-parameter reduction across 3 random observation matrices (2-DoF RRBot)")
    for trial in range(3):
        rows = []
        for _ in range(60):
            q = rng.uniform(-1.0, 1.0, kin.nDoF)
            dq = rng.uniform(-2.0, 2.0, kin.nDoF)
            ddq = rng.uniform(-3.0, 3.0, kin.nDoF)
            rows.append(newton_euler_regressor(kin, q, dq, ddq))
        W = np.vstack(rows)
        W_base, P, kept_cols, rank, pi_base = compute_base_parameters(W, pi)
        obs_err = np.max(np.abs(W @ pi - W_base @ pi_base))
        print(f"  Trial {trial+1}: W {W.shape} -> W_b {W_base.shape}, rank={rank}, max|W*pi - W_b*pi_b|={obs_err:.2e}")
        np.testing.assert_allclose(W @ pi, W_base @ pi_base, atol=1e-10)
        assert rank == W_base.shape[1]
        assert len(kept_cols) == rank
        assert P.shape[0] == rank
    print("  VERIFIED: Observation equation preserved across 3 random matrices (atol=1e-10)")


@pytest.mark.slow
def test_slow_condition_cost_matches_manual_base_matrix():
    from src.base_parameters import compute_base_parameters
    from src.dynamics_newton_euler import newton_euler_regressor
    from src.excitation import _condition_cost_base
    from src.kinematics import RobotKinematics
    from src.trajectory import build_frequencies, fourier_trajectory
    from src.urdf_parser import parse_urdf

    kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
    freqs = build_frequencies(0.2, 1)
    q0 = np.zeros(kin.nDoF)
    params = np.array([0.35, -0.25])
    t = np.linspace(0.0, 1.0 / 0.2, 101)
    q, dq, ddq = fourier_trajectory(
        params,
        freqs,
        t,
        q0,
        "cosine",
        False,
    )
    get_reg = lambda qv, dqv, ddqv: newton_euler_regressor(kin, qv, dqv, ddqv)

    cost = _condition_cost_base(q, dq, ddq, t, get_reg)

    step = max(1, t.size // 50)
    indices = list(range(0, t.size, step))
    W = np.vstack([get_reg(q[:, idx], dq[:, idx], ddq[:, idx]) for idx in indices])
    W_base, _, _, _, _ = compute_base_parameters(W, np.ones(W.shape[1]), tol=1e-8)
    sv = np.linalg.svd(W_base, compute_uv=False)
    sv_pos = sv[sv > 1e-12]
    manual = sv_pos[0] / sv_pos[-1]

    print("\nSTAGE 6 (slow): _condition_cost_base() must agree with manual SVD condition number")
    print(f"  _condition_cost_base() = {cost:.10f}")
    print(f"  manual SVD ratio       = {manual:.10f}")
    print(f"  |difference|           = {abs(cost - manual):.2e}")

    assert np.isfinite(cost)
    np.testing.assert_allclose(cost, manual, rtol=1e-10, atol=1e-10)
    print("  VERIFIED: Condition-cost function matches manual SVD computation (rtol=1e-10)")


@pytest.mark.slow
def test_slow_sc_3dof_fixture_remains_supported():
    from src.urdf_parser import parse_urdf

    robot = parse_urdf(URDF_3DOF)
    print(f"\nSTAGE 1 (slow): SC_3DoF fixture support")
    print(f"  nDoF = {robot.nDoF}")
    assert robot.nDoF == 3
    print("  VERIFIED: SC_3DoF URDF parses correctly")
