"""Pytest tests for the sysid_pipeline package."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy

# Ensure the sysid_pipeline root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_DIR = ASSET_DIR
URDF_RRBOT = str(URDF_DIR / "RRBot_single.urdf")
URDF_1DOF = str(URDF_DIR / "SC_1DoF.urdf")
URDF_3DOF = str(URDF_DIR / "SC_3DoF.urdf")
URDF_DEFAULT = URDF_RRBOT


# ──────────────────────────────────────────────────────────────────────────────
# 1. URDF chain extraction and Tw_0
# ──────────────────────────────────────────────────────────────────────────────

class TestURDFParsing:
    def test_default_rrbot_chain_and_tw0(self):
        from src.urdf_parser import parse_urdf
        r = parse_urdf(URDF_DEFAULT)
        assert r.nDoF == 2
        assert r.revolute_joint_names == ["single_rrbot_joint1", "single_rrbot_joint2"]
        np.testing.assert_allclose(r.Tw_0, np.eye(4), atol=1e-14)

    def test_1dof_chain_and_tw0(self):
        from src.urdf_parser import parse_urdf
        r = parse_urdf(URDF_1DOF)
        assert r.nDoF == 1
        np.testing.assert_allclose(r.Tw_0, np.eye(4), atol=1e-14)

    def test_3dof_chain_and_tw0(self):
        from src.urdf_parser import parse_urdf
        r = parse_urdf(URDF_3DOF)
        assert r.nDoF == 3
        assert r.Tw_0.shape == (4, 4)
        assert r.Tw_0[2, 3] != 0.0 or np.allclose(r.Tw_0, np.eye(4))

    def test_chain_order_is_topological(self):
        from src.urdf_parser import parse_urdf
        r = parse_urdf(URDF_3DOF)
        for i in range(len(r.chain_joints) - 1):
            assert r.chain_joints[i].child == r.chain_joints[i + 1].parent

    def test_revolute_names_match_chain(self):
        from src.urdf_parser import parse_urdf
        r = parse_urdf(URDF_3DOF)
        rev_from_chain = [j.name for j in r.chain_joints
                          if j.joint_type in ("revolute", "continuous")]
        assert r.revolute_joint_names == rev_from_chain


# ──────────────────────────────────────────────────────────────────────────────
# 2. Trajectory boundary conditions (A1.1)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrajectoryBoundary:
    @pytest.mark.parametrize("basis,opt_phase", [
        ("cosine", False),
        ("sine", False),
        ("both", False),
        ("both", True),
    ])
    def test_initial_conditions(self, basis, opt_phase):
        from src.trajectory import fourier_trajectory, build_frequencies
        nDoF, m = 3, 3
        freqs = build_frequencies(0.5, m)
        q0 = np.array([0.1, -0.5, 0.3])
        np.random.seed(42)
        n_params = nDoF * m if basis in ("cosine", "sine") else nDoF * 2 * m
        params = np.random.uniform(-0.3, 0.3, n_params)
        t = np.linspace(0, 2.0, 500)
        q, dq, _ = fourier_trajectory(params, freqs, t, q0, basis, opt_phase)
        np.testing.assert_allclose(q[:, 0], q0, atol=1e-14)
        np.testing.assert_allclose(dq[:, 0], 0.0, atol=1e-14)

    def test_cosine_periodic_velocity(self):
        """For cosine basis with integer-period T, dq(T) should be ~0."""
        from src.trajectory import fourier_trajectory, build_frequencies
        nDoF, m = 2, 3
        freqs = build_frequencies(0.5, m)
        q0 = np.zeros(nDoF)
        np.random.seed(123)
        params = np.random.uniform(-0.2, 0.2, nDoF * m)
        T = 1.0 / 0.5  # one period
        t = np.linspace(0, T, 1000)
        _, dq, _ = fourier_trajectory(params, freqs, t, q0, "cosine", False)
        np.testing.assert_allclose(dq[:, -1], 0.0, atol=1e-12)

    def test_sine_integer_period_endpoint_velocity(self):
        """For sine basis with integer periods, dq(T) must be zero."""
        from src.trajectory import fourier_trajectory, build_frequencies
        nDoF, m = 2, 3
        freqs = build_frequencies(0.5, m)
        q0 = np.zeros(nDoF)
        np.random.seed(99)
        params = np.random.uniform(-0.2, 0.2, nDoF * m)
        T = 1.0 / 0.5  # one base period
        t = np.linspace(0, T, 1000)
        _, dq, ddq = fourier_trajectory(params, freqs, t, q0, "sine", False)
        np.testing.assert_allclose(dq[:, -1], 0.0, atol=1e-12)
        np.testing.assert_allclose(ddq[:, -1], 0.0, atol=1e-12)

    def test_sine_noninteger_period_rejected_by_config(self):
        """Config validation must reject sine with non-integer periods."""
        from src.config_loader import load_config
        cfg_data = {
            "urdf_path": URDF_1DOF,
            "output_dir": "dummy",
            "excitation": {
                "basis_functions": "sine",
                "trajectory_duration_periods": 1.5,
            },
        }
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump(cfg_data, f)
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="integer"):
                load_config(tmp)
        finally:
            os.unlink(tmp)

    def test_sine_integer_period_accepted_by_config(self):
        """Config validation must accept sine with integer periods."""
        from src.config_loader import load_config
        cfg_data = {
            "urdf_path": URDF_1DOF,
            "output_dir": "dummy",
            "excitation": {
                "basis_functions": "sine",
                "trajectory_duration_periods": 2,
            },
        }
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump(cfg_data, f)
            tmp = f.name
        try:
            cfg = load_config(tmp)  # should not raise
            assert cfg["excitation"]["trajectory_duration_periods"] == 2
        finally:
            os.unlink(tmp)

    def test_el_with_constrained_identification_rejected(self):
        """EL + lmi/cholesky must be rejected at config validation."""
        from src.config_loader import load_config
        import tempfile, os
        for feas in ("lmi", "cholesky"):
            cfg_data = {
                "urdf_path": URDF_1DOF,
                "output_dir": "dummy",
                "method": "euler_lagrange",
                "identification": {"feasibility_method": feas},
            }
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                             delete=False) as f:
                json.dump(cfg_data, f)
                tmp = f.name
            try:
                with pytest.raises(ValueError, match="euler_lagrange"):
                    load_config(tmp)
            finally:
                os.unlink(tmp)

    def test_cholesky_accepted_as_distinct_method(self):
        """Config validation accepts 'cholesky' as a distinct feasibility method."""
        from src.config_loader import load_config
        import tempfile, os, warnings
        cfg_data = {
            "urdf_path": URDF_1DOF,
            "output_dir": "dummy",
            "method": "newton_euler",
            "identification": {"feasibility_method": "cholesky"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False) as f:
            json.dump(cfg_data, f)
            tmp = f.name
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                cfg = load_config(tmp)
                assert cfg["identification"]["feasibility_method"] == "cholesky"
                dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(dep_warnings) == 0
        finally:
            os.unlink(tmp)


# ──────────────────────────────────────────────────────────────────────────────
# 3. NE / EL torque equivalence
# ──────────────────────────────────────────────────────────────────────────────

class TestNEvsEL:
    @pytest.fixture
    def kin_default(self):
        from src.urdf_parser import parse_urdf
        from src.kinematics import RobotKinematics
        return RobotKinematics(parse_urdf(URDF_DEFAULT))

    @pytest.fixture
    def kin_3dof(self):
        from src.urdf_parser import parse_urdf
        from src.kinematics import RobotKinematics
        return RobotKinematics(parse_urdf(URDF_3DOF))

    @pytest.fixture
    def kin_1dof(self):
        from src.urdf_parser import parse_urdf
        from src.kinematics import RobotKinematics
        return RobotKinematics(parse_urdf(URDF_1DOF))

    def _compare(self, kin, cache_dir, n_states=5):
        import shutil
        from src.dynamics_newton_euler import newton_euler_regressor
        from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder
        cache = Path(cache_dir)
        if cache.exists():
            shutil.rmtree(cache)
        reg_fn, kept = euler_lagrange_regressor_builder(kin, str(cache))
        pi = kin.PI.flatten()
        rng = np.random.default_rng(42)
        for _ in range(n_states):
            q = rng.uniform(-1.5, 1.5, kin.nDoF)
            dq = rng.uniform(-2, 2, kin.nDoF)
            ddq = rng.uniform(-3, 3, kin.nDoF)
            Y_NE = newton_euler_regressor(kin, q, dq, ddq)
            Y_EL = reg_fn(q, dq, ddq)
            tau_NE = Y_NE @ pi
            tau_EL = Y_EL @ pi[kept]
            np.testing.assert_allclose(tau_NE, tau_EL, atol=1e-10)

    def test_default_rrbot(self, kin_default, tmp_path):
        self._compare(kin_default, str(tmp_path / "el_cache_default"))

    def test_builder_succeeds_with_deprecation_warnings_as_errors(self, kin_default, tmp_path):
        from src.dynamics_euler_lagrange import euler_lagrange_regressor_builder

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            reg_fn, kept = euler_lagrange_regressor_builder(
                kin_default, str(tmp_path / "el_cache_warning_guard")
            )

        assert callable(reg_fn)
        assert len(kept) > 0

    def test_1dof(self, kin_1dof, tmp_path):
        self._compare(kin_1dof, str(tmp_path / "el_cache_1"))

    @pytest.mark.slow
    def test_3dof(self, kin_3dof, tmp_path):
        self._compare(kin_3dof, str(tmp_path / "el_cache_3"))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Base parameter reduction consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestBaseParameters:
    def test_observation_equation_preserved(self):
        from src.urdf_parser import parse_urdf
        from src.kinematics import RobotKinematics
        from src.dynamics_newton_euler import newton_euler_regressor
        from src.base_parameters import compute_base_parameters

        kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
        rng = np.random.default_rng(99)
        rows = []
        for _ in range(80):
            q = rng.uniform(-1, 1, kin.nDoF)
            dq = rng.uniform(-2, 2, kin.nDoF)
            ddq = rng.uniform(-3, 3, kin.nDoF)
            rows.append(newton_euler_regressor(kin, q, dq, ddq))
        W = np.vstack(rows)
        pi = kin.PI.flatten()

        W_base, P, kept, rank, pi_base = compute_base_parameters(W, pi)
        np.testing.assert_allclose(W @ pi, W_base @ pi_base, atol=1e-10)

    def test_reconstruction_via_pinv(self):
        from src.urdf_parser import parse_urdf
        from src.kinematics import RobotKinematics
        from src.dynamics_newton_euler import newton_euler_regressor
        from src.base_parameters import compute_base_parameters

        kin = RobotKinematics(parse_urdf(URDF_DEFAULT))
        rng = np.random.default_rng(77)
        rows = []
        for _ in range(80):
            q = rng.uniform(-1, 1, kin.nDoF)
            dq = rng.uniform(-2, 2, kin.nDoF)
            ddq = rng.uniform(-3, 3, kin.nDoF)
            rows.append(newton_euler_regressor(kin, q, dq, ddq))
        W = np.vstack(rows)
        pi = kin.PI.flatten()

        W_base, P, kept, rank, pi_base = compute_base_parameters(W, pi)
        pi_recon = np.linalg.pinv(P) @ pi_base
        np.testing.assert_allclose(W @ pi_recon, W @ pi, atol=1e-10)


class TestEulerLagrangeStructuralPruning:
    def test_remove_zero_columns_prunes_only_structural_zeros(self):
        from src.dynamics_euler_lagrange import _remove_zero_columns

        x = sympy.Symbol("x")
        Y = sympy.Matrix([
            [sympy.S.Zero, x, sympy.sin(1)**2 + sympy.cos(1)**2 - 1],
            [0, 2 * x, 0],
        ])

        reduced, kept = _remove_zero_columns(Y)

        assert kept == [1, 2]
        assert reduced.shape == (2, 2)
        assert reduced[:, 0] == Y[:, 1]
        assert reduced[:, 1] == Y[:, 2]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Filtering / downsampling
# ──────────────────────────────────────────────────────────────────────────────

class TestFiltering:
    def test_passthrough_when_disabled(self):
        from src.filtering import apply_filter
        x = np.random.randn(100, 3)
        y = apply_filter(x, 1000.0, {"enabled": False})
        np.testing.assert_array_equal(x, y)

    def test_lowpass_attenuates_high_freq(self):
        from src.filtering import apply_filter
        fs = 1000.0
        t = np.arange(0, 1, 1.0 / fs)
        low = np.sin(2 * np.pi * 5 * t)
        high = np.sin(2 * np.pi * 200 * t)
        signal = (low + high).reshape(-1, 1)
        cfg = {"enabled": True, "cutoff_frequency_hz": 50.0, "filter_order": 4}
        filtered = apply_filter(signal, fs, cfg)
        high_power_before = np.var(high)
        high_residual = filtered.flatten() - low
        high_power_after = np.var(high_residual)
        assert high_power_after < 0.01 * high_power_before


# ──────────────────────────────────────────────────────────────────────────────
# 6. Pseudo-inertia physical feasibility
# ──────────────────────────────────────────────────────────────────────────────

class TestFeasibility:
    def test_physically_valid_params_pass(self):
        """Parameters derived from a real body (CoM at origin) must pass."""
        from src.feasibility import check_feasibility, is_pseudo_inertia_psd
        pi = np.array([2.0, 0, 0, 0, 0.1, 0, 0, 0.1, 0, 0.1])
        assert is_pseudo_inertia_psd(pi)
        report, feasible, _ = check_feasibility(pi, 1)
        assert feasible

    def test_negative_mass_fails(self):
        from src.feasibility import check_feasibility
        pi = np.array([-1.0, 0, 0, 0, 0.1, 0, 0, 0.1, 0, 0.1])
        report, feasible, _ = check_feasibility(pi, 1)
        assert not feasible
        assert any("Pseudo-inertia" in issue or "mass" in issue
                    for issue in report[0]["issues"])

    def test_large_first_moment_vs_tiny_mass_fails_pseudo_inertia(self):
        """Inertia PSD + positive mass + triangle OK, but J not PSD.

        This is the case described in the task: mass ≈ 0, first moments
        large → J has negative eigenvalue → not physically realisable.
        """
        from src.feasibility import (check_feasibility, is_pseudo_inertia_psd,
                                      pseudo_inertia_matrix)
        # tiny mass, large first moments, inertia that passes old checks
        pi = np.array([1e-6, 0.255, 0.262, 0.0, 0.1, 0, 0, 0.1, 0, 0.1])
        # Old-style checks would pass (mass>0, I PSD, triangle OK)
        I_mat = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        assert np.all(np.linalg.eigvalsh(I_mat) > 0), "Inertia is PSD"
        # But pseudo-inertia must fail
        J = pseudo_inertia_matrix(pi)
        J_eigs = np.linalg.eigvalsh(J)
        assert np.min(J_eigs) < 0, "J must have negative eigenvalue"
        assert not is_pseudo_inertia_psd(pi)
        report, feasible, _ = check_feasibility(pi, 1)
        assert not feasible
        assert any("Pseudo-inertia" in issue for issue in report[0]["issues"])

    def test_projection_produces_psd_pseudo_inertia(self):
        """After LMI projection, corrected parameters must have J ≽ 0."""
        from src.feasibility import check_feasibility, is_pseudo_inertia_psd
        pi_bad = np.array([1e-6, 0.255, 0.262, 0.0, 0.1, 0, 0, 0.1, 0, 0.1])
        report, feasible, pi_corrected = check_feasibility(pi_bad, 1, method="lmi")
        assert not feasible  # original is bad
        assert is_pseudo_inertia_psd(pi_corrected)  # corrected must be good

    def test_projection_produces_psd_cholesky(self):
        """Cholesky projection also produces J ≽ 0."""
        from src.feasibility import check_feasibility, is_pseudo_inertia_psd
        pi_bad = np.array([2.0, 0, 0, 0, -0.1, 0, 0, 0.1, 0, 0.1])
        report, feasible, pi_corrected = check_feasibility(pi_bad, 1, method="cholesky")
        assert not feasible
        assert is_pseudo_inertia_psd(pi_corrected)

    def test_known_good_body_pseudo_inertia(self):
        """A unit cube (m=1, uniform density) must be physically valid."""
        from src.feasibility import is_pseudo_inertia_psd
        # CoM at origin, I = diag(1/6, 1/6, 1/6) for unit cube
        I_val = 1.0 / 6.0
        pi = np.array([1.0, 0, 0, 0, I_val, 0, 0, I_val, 0, I_val])
        assert is_pseudo_inertia_psd(pi)

    def test_pi_from_pseudo_inertia_roundtrip(self):
        """pi → J → pi must recover the original parameter vector."""
        from src.feasibility import pseudo_inertia_matrix, pi_from_pseudo_inertia_matrix
        pi = np.array([2.5, 0.1, -0.2, 0.05, 0.3, 0.01, -0.02, 0.25, 0.005, 0.35])
        J = pseudo_inertia_matrix(pi)
        pi_back = pi_from_pseudo_inertia_matrix(J)
        np.testing.assert_allclose(pi_back, pi, atol=1e-14)

    def test_cholesky_solver_guarantees_psd(self):
        """Parameters produced by the Cholesky solver path must have J ≽ 0."""
        from src.solver import _solve_cholesky, _cholesky_vec_to_lower, _lower_to_cholesky_vec
        from src.feasibility import pseudo_inertia_matrix, is_pseudo_inertia_psd
        nDoF = 1
        # Construct a small synthetic problem
        rng = np.random.default_rng(42)
        W = rng.standard_normal((30, 10))
        tau = rng.standard_normal(30)
        P_mat = np.eye(10)
        pi_hat, res, info = _solve_cholesky(W, tau, nDoF, "ols", None, P_mat)
        assert info["solved_in_full_space"]
        assert is_pseudo_inertia_psd(pi_hat[:10])


# ──────────────────────────────────────────────────────────────────────────────
# 7. Insufficient sample handling
# ──────────────────────────────────────────────────────────────────────────────

class TestSampleSufficiency:
    def test_insufficient_samples_raises(self):
        """build_observation_matrix must raise when equations < unknowns."""
        from src.observation_matrix import build_observation_matrix
        from src.urdf_parser import parse_urdf
        from src.kinematics import RobotKinematics
        from src.dynamics_newton_euler import newton_euler_regressor

        kin = RobotKinematics(parse_urdf(URDF_3DOF))
        # Only 1 time sample for 3-DoF (3 equations for ~19+ unknowns)
        q = np.zeros((1, 3))
        dq = np.zeros((1, 3))
        ddq = np.zeros((1, 3))
        tau = np.zeros((1, 3))

        def reg_fn(q_v, dq_v, ddq_v):
            return newton_euler_regressor(kin, q_v, dq_v, ddq_v)

        cfg = {
            "friction": {"model": "none"},
            "filtering": {"enabled": False},
            "downsampling": {"frequency_hz": 0},
        }

        with pytest.raises(ValueError, match="Insufficient data"):
            build_observation_matrix(q, dq, ddq, tau, reg_fn, cfg, 100.0)


# ──────────────────────────────────────────────────────────────────────────────
# 8. End-to-end pipeline smoke
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineSmoke:
    def _make_config(self, tmp_path, urdf, method="newton_euler",
                     n_dof=None, feasibility="none"):
        from src.urdf_parser import parse_urdf
        if n_dof is None:
            n_dof = parse_urdf(urdf).nDoF
        cfg = {
            "urdf_path": urdf,
            "output_dir": str(tmp_path / "out"),
            "method": method,
            "joint_limits": {
                "position": [[-1, 1]] * n_dof,
                "velocity": [[-2, 2]] * n_dof,
                "acceleration": [[-5, 5]] * n_dof,
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
                "parameter_bounds": False,
                "feasibility_method": feasibility,
                "data_file": None,
            },
            "filtering": {"enabled": False},
            "downsampling": {"frequency_hz": 0},
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(cfg))
        return str(cfg_path)

    def test_ne_default_rrbot(self, tmp_path):
        from src.pipeline import SystemIdentificationPipeline
        cfg = self._make_config(tmp_path, URDF_DEFAULT)
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert results["residual"] < 1e-8

    def test_el_default_rrbot(self, tmp_path):
        from src.pipeline import SystemIdentificationPipeline
        cfg = self._make_config(tmp_path, URDF_DEFAULT, method="euler_lagrange")
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert results["residual"] < 1e-8

    def test_ne_1dof(self, tmp_path):
        from src.pipeline import SystemIdentificationPipeline
        cfg = self._make_config(tmp_path, URDF_1DOF, n_dof=1)
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert results["residual"] < 1e-8

    def test_ne_3dof(self, tmp_path):
        from src.pipeline import SystemIdentificationPipeline
        cfg = self._make_config(tmp_path, URDF_3DOF, n_dof=3)
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert results["residual"] < 1e-6

    def test_lmi_constrained_produces_feasible_model(self, tmp_path):
        """Constrained identification with LMI must produce pseudo-inertia PSD result."""
        from src.pipeline import SystemIdentificationPipeline
        from src.feasibility import is_pseudo_inertia_psd
        cfg = self._make_config(tmp_path, URDF_DEFAULT, feasibility="lmi")
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert float(results["residual"]) < 1.0
        pi = results["pi_identified"]
        # The identified model or its correction must pass pseudo-inertia PSD
        pi_corrected = results["pi_corrected"]
        if len(pi_corrected) >= 10:
            assert is_pseudo_inertia_psd(pi_corrected[:10])

    def test_cholesky_constrained_produces_feasible_model(self, tmp_path):
        """Cholesky reparameterisation must produce pseudo-inertia PSD result."""
        from src.pipeline import SystemIdentificationPipeline
        from src.feasibility import is_pseudo_inertia_psd
        cfg = self._make_config(tmp_path, URDF_DEFAULT, feasibility="cholesky")
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert float(results["residual"]) < 1.0
        pi_corrected = results["pi_corrected"]
        if len(pi_corrected) >= 10:
            assert is_pseudo_inertia_psd(pi_corrected[:10])

    @pytest.mark.parametrize("urdf,n_dof", [
        (URDF_1DOF, 1),
        (URDF_3DOF, 3),
    ])
    def test_ne_sc_models_remain_functional(self, tmp_path, urdf, n_dof):
        from src.pipeline import SystemIdentificationPipeline
        cfg = self._make_config(tmp_path, urdf, n_dof=n_dof)
        pipe = SystemIdentificationPipeline(cfg)
        pipe.run()
        results = np.load(str(tmp_path / "out" / "identification_results.npz"),
                          allow_pickle=True)
        assert results["residual"] < 1e-6
