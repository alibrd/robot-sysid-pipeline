"""Tests for the dynamics-model runtime helpers and the closed-form emitter.

Two groups:

  * Runtime helpers (``src.dynamics_model``): mathematical identities for
    M, c, g, tau_f at machine precision.
  * Emitted closed-form module (``src.regressor_export.export_dynamics_model_closed_form``):
    end-to-end correctness against the in-process helpers and the
    linear-in-parameter regressor.

Default fixture: ``tests/assets/RRBot_single.urdf`` (2-DOF). Test #17 also
exercises the 3-DOF ``ElbowManipulator_3DoF.urdf`` to confirm the mode
1 vs mode 2 file content really differs.
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URDF_RRBOT = ROOT / "tests" / "assets" / "RRBot_single.urdf"
URDF_ELBOW = ROOT / "tests" / "assets" / "ElbowManipulator_3DoF.urdf"


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _build_model(urdf, friction_model="none", backend="newton_euler",
                 cache_dir=None):
    from src.regressor_model import RegressorModel
    return RegressorModel.from_urdf(
        urdf,
        friction_model=friction_model,
        backend=backend,
        cache_dir=cache_dir,
    )


def _random_state(n, seed, *, amp_dq=0.5, amp_ddq=1.0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-0.5, 0.5, n)
    dq = rng.uniform(-amp_dq, amp_dq, n)
    ddq = rng.uniform(-amp_ddq, amp_ddq, n)
    return q, dq, ddq


def _load_emitted(path):
    spec = importlib.util.spec_from_file_location(
        f"_emitted_dyn_{path.stem}_{id(path)}", str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_pi_aug(model, *, seed=0, friction_scale=0.1):
    rng = np.random.default_rng(seed)
    pi_rigid = model.kin.PI.flatten().copy()
    n_fric = model.n_friction_params
    pi_fric = rng.uniform(0.0, friction_scale, n_fric)
    return np.concatenate([pi_rigid, pi_fric])


# ---------------------------------------------------------------------------
# Group 1: runtime helpers (mathematical identities)
# ---------------------------------------------------------------------------

def test_gravity_dq_ddq_zero(tmp_path):
    from src.dynamics_model import gravity_vector
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()
    q, _, _ = _random_state(model.nDoF, 0)
    g_ref = model.rigid(q, np.zeros(model.nDoF), np.zeros(model.nDoF)) @ pi_rigid
    g_q = gravity_vector(model.rigid, pi_rigid, q)
    np.testing.assert_allclose(g_q, g_ref, atol=1e-13)


def test_mass_matrix_symmetric_runtime(tmp_path):
    from src.dynamics_model import mass_matrix
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()
    for seed in range(3):
        q, _, _ = _random_state(model.nDoF, seed)
        M = mass_matrix(model.rigid, pi_rigid, q)
        assert np.max(np.abs(M - M.T)) < 1e-10, f"asymmetric M at seed {seed}"


def test_mass_matrix_positive_definite_runtime(tmp_path):
    from src.dynamics_model import mass_matrix
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()  # URDF inertials guaranteed PD
    for seed in range(3):
        q, _, _ = _random_state(model.nDoF, seed)
        M = mass_matrix(model.rigid, pi_rigid, q)
        Msym = 0.5 * (M + M.T)
        eigs = np.linalg.eigvalsh(Msym)
        assert np.all(eigs > 0), f"M not PD at seed {seed}, eigs={eigs}"


def test_coriolis_vector_random(tmp_path):
    from src.dynamics_model import coriolis_vector, gravity_vector
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()
    for seed in range(3):
        q, dq, _ = _random_state(model.nDoF, seed)
        c = coriolis_vector(model.rigid, pi_rigid, q, dq)
        g_q = gravity_vector(model.rigid, pi_rigid, q)
        ref = model.rigid(q, dq, np.zeros(model.nDoF)) @ pi_rigid - g_q
        np.testing.assert_allclose(c, ref, atol=1e-12)


def test_coriolis_matrix_christoffel_product_runtime(tmp_path):
    from src.dynamics_model import (
        coriolis_matrix_christoffel,
        coriolis_vector,
    )
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()
    for seed in range(3):
        q, dq, _ = _random_state(model.nDoF, seed)
        C_mat = coriolis_matrix_christoffel(model.rigid, pi_rigid, q, dq)
        c = coriolis_vector(model.rigid, pi_rigid, q, dq)
        np.testing.assert_allclose(C_mat @ dq, c, atol=1e-7)


def test_coriolis_matrix_christoffel_skew_symmetry(tmp_path):
    """dM/dt - 2C must be skew-symmetric for the true Christoffel C.

    Regression test: a wrong axis permutation in the Christoffel assembly
    still satisfies c = C @ dq (the error symmetrises out over the j,k
    contraction) but breaks this passivity property.
    """
    from src.dynamics_model import coriolis_matrix_christoffel, mass_matrix
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()
    n = model.nDoF
    h = 1e-6
    for seed in range(3):
        q, dq, _ = _random_state(n, seed)
        C_mat = coriolis_matrix_christoffel(model.rigid, pi_rigid, q, dq)
        M_dot = np.zeros((n, n))
        for k in range(n):
            qp = q.copy()
            qp[k] += h
            qm = q.copy()
            qm[k] -= h
            M_dot += dq[k] * (
                mass_matrix(model.rigid, pi_rigid, qp)
                - mass_matrix(model.rigid, pi_rigid, qm)
            ) / (2.0 * h)
        N = M_dot - 2.0 * C_mat
        skew_err = np.max(np.abs(N + N.T))
        assert skew_err < 1e-6, (
            f"dM/dt - 2C not skew-symmetric (err={skew_err:.3e}) at seed {seed}"
        )


def test_dynamics_identity_no_friction(tmp_path):
    from src.dynamics_model import verify_dynamics_consistency
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi_rigid = model.kin.PI.flatten()
    for seed in range(5):
        q, dq, ddq = _random_state(model.nDoF, seed)
        _, _, err = verify_dynamics_consistency(
            model.rigid, pi_rigid, q, dq, ddq,
        )
        assert err < 1e-12, f"identity mismatch {err:.3e} at seed {seed}"


def test_dynamics_identity_with_friction_viscous_coulomb(tmp_path):
    from src.dynamics_model import verify_dynamics_consistency
    model = _build_model(URDF_RRBOT, friction_model="viscous_coulomb",
                         cache_dir=tmp_path)
    pi_aug = _make_pi_aug(model, seed=11)
    n_rigid = 10 * model.nDoF
    pi_rigid = pi_aug[:n_rigid]
    pi_friction = pi_aug[n_rigid:]
    for seed in range(5):
        q, dq, ddq = _random_state(model.nDoF, seed)
        _, _, err = verify_dynamics_consistency(
            model.rigid, pi_rigid, q, dq, ddq,
            pi_friction=pi_friction, friction_model="viscous_coulomb",
        )
        assert err < 1e-12, f"identity mismatch {err:.3e} at seed {seed}"


def test_friction_torque_signature_and_invariance(tmp_path):
    from src.dynamics_model import friction_torque
    pi_friction = np.array([0.3, 0.5])
    dq = np.array([0.7, -0.4])
    tau_f = friction_torque(dq, pi_friction, "viscous")
    np.testing.assert_allclose(tau_f, pi_friction * dq, atol=1e-15)


def test_friction_zero_velocity_viscous(tmp_path):
    from src.dynamics_model import friction_torque
    model = _build_model(URDF_RRBOT, friction_model="viscous",
                         cache_dir=tmp_path)
    n = model.nDoF
    pi_friction = np.array([0.3, 0.5])
    tau_f = friction_torque(np.zeros(n), pi_friction, "viscous")
    np.testing.assert_array_equal(tau_f, np.zeros(n))


def test_ne_el_cross_consistency_runtime(tmp_path):
    """NE vs EL backends agree on M, c, g, tau_f at machine precision."""
    from src.dynamics_model import compute_full_dynamics
    cache_ne = tmp_path / "ne"
    cache_el = tmp_path / "el"
    model_ne = _build_model(URDF_RRBOT, friction_model="viscous_coulomb",
                            backend="newton_euler", cache_dir=cache_ne)
    model_el = _build_model(URDF_RRBOT, friction_model="viscous_coulomb",
                            backend="euler_lagrange", cache_dir=cache_el)
    pi_aug = _make_pi_aug(model_ne, seed=3)
    n_rigid = 10 * model_ne.nDoF
    pi_rigid = pi_aug[:n_rigid]
    pi_friction = pi_aug[n_rigid:]
    for seed in range(3):
        q, dq, _ = _random_state(model_ne.nDoF, seed)
        d_ne = compute_full_dynamics(
            model_ne.rigid, pi_rigid, q, dq,
            pi_friction=pi_friction, friction_model="viscous_coulomb",
        )
        d_el = compute_full_dynamics(
            model_el.rigid, pi_rigid, q, dq,
            pi_friction=pi_friction, friction_model="viscous_coulomb",
        )
        for k in ("g", "M", "c", "tau_f"):
            np.testing.assert_allclose(
                d_ne[k], d_el[k], atol=1e-8,
                err_msg=f"NE vs EL mismatch in {k}, seed {seed}",
            )


# ---------------------------------------------------------------------------
# Group 2: emitted closed-form module
# ---------------------------------------------------------------------------

def _emit_for(model, pi_aug, tmp_path, *, simplify="trigsimp",
              include_coriolis_matrix=False):
    from src.regressor_export import export_dynamics_model_closed_form
    return export_dynamics_model_closed_form(
        model, pi_aug=pi_aug, output_dir=tmp_path, simplify=simplify,
        include_coriolis_matrix=include_coriolis_matrix,
    )


def test_emitted_module_loads_standalone(tmp_path):
    model = _build_model(URDF_RRBOT, friction_model="viscous",
                         cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=1)
    path = _emit_for(model, pi, tmp_path)
    mod = _load_emitted(path)
    for name in ("g", "M", "c", "tau_f", "tau"):
        assert hasattr(mod, name), f"missing function {name}"


def test_emitted_no_parameters_pkl_dependency(tmp_path):
    model = _build_model(URDF_RRBOT, friction_model="viscous_coulomb",
                         cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=2)
    path = _emit_for(model, pi, tmp_path)
    text = path.read_text(encoding="utf-8")
    for banned in ("parameters.pkl", "import pickle", "from pickle",
                   "import regressor", "from regressor",
                   "from src", "import src"):
        assert banned not in text, f"banned token in emitted source: {banned!r}"


def test_emitted_M_is_standalone(tmp_path):
    """The body of M (and g/c/tau_f) must not call sibling functions."""
    model = _build_model(URDF_RRBOT, friction_model="viscous",
                         cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=4)
    path = _emit_for(model, pi, tmp_path)
    text = path.read_text(encoding="utf-8")

    def _body_of(fn_name):
        # Capture the function body until the next top-level "def " or EOF.
        m = re.search(rf"^def {fn_name}\(.*?\):\n", text, flags=re.MULTILINE)
        assert m, f"function {fn_name} not found in emitted source"
        start = m.end()
        nxt = re.search(r"^def\s", text[start:], flags=re.MULTILINE)
        return text[start:start + (nxt.start() if nxt else len(text))]

    callers = ("g(", "M(", "c(", "tau_f(", "Y_rigid(", "Y_augmented(",
               "Y_friction(", "tau(")
    for fn in ("g", "M", "c", "tau_f"):
        body = _body_of(fn)
        for caller in callers:
            # Skip the self-recursion check for the same name (irrelevant).
            if caller.startswith(fn + "("):
                continue
            assert caller not in body, (
                f"emitted {fn}() body calls {caller}, expected fully "
                f"standalone closed-form expression"
            )


def test_emitted_matches_runtime(tmp_path):
    from src.dynamics_model import (
        gravity_vector, mass_matrix, coriolis_vector, friction_torque,
    )
    model = _build_model(URDF_RRBOT, friction_model="viscous_coulomb",
                         cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=5)
    n_rigid = 10 * model.nDoF
    pi_rigid = pi[:n_rigid]
    pi_friction = pi[n_rigid:]
    path = _emit_for(model, pi, tmp_path)
    mod = _load_emitted(path)
    for seed in range(50):
        q, dq, _ = _random_state(model.nDoF, seed)
        np.testing.assert_allclose(
            mod.g(q),
            gravity_vector(model.rigid, pi_rigid, q),
            atol=1e-10, err_msg=f"g mismatch seed {seed}",
        )
        np.testing.assert_allclose(
            mod.M(q),
            mass_matrix(model.rigid, pi_rigid, q),
            atol=1e-10, err_msg=f"M mismatch seed {seed}",
        )
        np.testing.assert_allclose(
            mod.c(q, dq),
            coriolis_vector(model.rigid, pi_rigid, q, dq),
            atol=1e-10, err_msg=f"c mismatch seed {seed}",
        )
        np.testing.assert_allclose(
            mod.tau_f(dq),
            friction_torque(dq, pi_friction, "viscous_coulomb"),
            atol=1e-10, err_msg=f"tau_f mismatch seed {seed}",
        )


def test_emitted_matches_regressor_tau(tmp_path):
    from src.friction import build_friction_regressor
    model = _build_model(URDF_RRBOT, friction_model="viscous_coulomb",
                         cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=6)
    path = _emit_for(model, pi, tmp_path)
    mod = _load_emitted(path)
    for seed in range(20):
        q, dq, ddq = _random_state(model.nDoF, seed)
        Y_aug = np.hstack((
            model.rigid(q, dq, ddq),
            build_friction_regressor(dq, "viscous_coulomb"),
        ))
        tau_linear = Y_aug @ pi
        tau_em = mod.tau(q, dq, ddq)
        err = float(np.max(np.abs(tau_linear - tau_em)))
        assert err < 1e-9, f"tau mismatch {err:.3e} at seed {seed}"


def test_emitted_M_symmetric(tmp_path):
    model = _build_model(URDF_RRBOT, friction_model="viscous",
                         cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=7)
    path = _emit_for(model, pi, tmp_path)
    mod = _load_emitted(path)
    for seed in range(5):
        q, _, _ = _random_state(model.nDoF, seed)
        M = mod.M(q)
        assert np.max(np.abs(M - M.T)) < 1e-10


def test_coriolis_matrix_runtime_matches_emitted(tmp_path):
    """Runtime Christoffel C agrees with the emitted closed-form C.

    Regression test: the emitted C is derived symbolically from the correct
    Christoffel formula; the runtime finite-difference C must match it.
    """
    from src.dynamics_model import coriolis_matrix_christoffel
    model = _build_model(URDF_RRBOT, cache_dir=tmp_path)
    pi = _make_pi_aug(model, seed=9)
    pi_rigid = pi[:10 * model.nDoF]
    path = _emit_for(model, pi, tmp_path, include_coriolis_matrix=True)
    mod = _load_emitted(path)
    for seed in range(3):
        q, dq, _ = _random_state(model.nDoF, seed)
        C_rt = coriolis_matrix_christoffel(model.rigid, pi_rigid, q, dq)
        np.testing.assert_allclose(
            mod.C(q, dq), C_rt, atol=1e-6,
            err_msg=f"runtime vs emitted C mismatch at seed {seed}",
        )


def test_emitted_mode1_vs_mode2_have_different_baked_pi(tmp_path):
    model = _build_model(URDF_ELBOW, friction_model="viscous_coulomb",
                         cache_dir=tmp_path)
    n = model.nDoF
    n_rigid = 10 * n
    n_aug = model.n_augmented_params

    # Mode 1: URDF nominal + zero friction
    pi_nominal = np.concatenate([
        model.kin.PI.flatten(),
        np.zeros(model.n_friction_params),
    ])
    # Mode 2: perturbed
    rng = np.random.default_rng(8)
    pi_identified = pi_nominal + 0.05 * rng.standard_normal(n_aug)
    pi_identified[n_rigid:] = np.abs(pi_identified[n_rigid:])  # keep physical

    out1 = tmp_path / "mode1"
    out1.mkdir()
    out2 = tmp_path / "mode2"
    out2.mkdir()
    from src.regressor_export import export_dynamics_model_closed_form
    p1 = export_dynamics_model_closed_form(model, pi_nominal, out1)
    p2 = export_dynamics_model_closed_form(model, pi_identified, out2)
    assert p1.read_text(encoding="utf-8") != p2.read_text(encoding="utf-8"), (
        "Mode 1 and Mode 2 emitted files are byte-identical despite "
        "different baked parameters"
    )

    mod1 = _load_emitted(p1)
    mod2 = _load_emitted(p2)
    # Mode 1: tau_f returns zeros for any dq.
    for seed in range(3):
        _, dq, _ = _random_state(n, seed)
        np.testing.assert_allclose(
            mod1.tau_f(dq), np.zeros(n), atol=1e-12,
            err_msg=f"mode1 tau_f not zero, seed={seed}",
        )
    # Mode 2: tau_f is non-trivial for general dq.
    for seed in range(3):
        _, dq, _ = _random_state(n, seed + 100)
        tau_f2 = mod2.tau_f(dq)
        assert np.max(np.abs(tau_f2)) > 0.0, (
            f"mode2 tau_f unexpectedly zero, seed={seed}, dq={dq}"
        )


def test_emitted_pipeline_mode1_excitation_only(tmp_path):
    """End-to-end Mode 1 smoke: excitation-only run emits dynamics_model.py."""
    from src.pipeline import SystemIdentificationPipeline
    n = 2
    cfg = {
        "urdf_path": str(URDF_RRBOT),
        "output_dir": str(tmp_path / "out"),
        "method": "newton_euler",
        "excitation_only": True,
        "friction": {"model": "viscous"},
        "identification": {"solver": "ols", "feasibility_method": "none"},
        "joint_limits": {
            "position": [[-3.14, 3.14]] * n,
            "velocity": [[-3.0, 3.0]] * n,
            "acceleration": [[-8.0, 8.0]] * n,
        },
        "excitation": {
            "num_harmonics": 3,
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 1,
            "optimize_condition_number": True,
            "optimizer_max_iter": 50,
        },
    }
    SystemIdentificationPipeline(cfg).run()
    out = Path(cfg["output_dir"])
    dyn_path = out / "dynamics_model.py"
    assert dyn_path.exists()
    mod = _load_emitted(dyn_path)
    # Mode 1 → zero friction
    for seed in range(3):
        _, dq, _ = _random_state(n, seed)
        np.testing.assert_allclose(
            mod.tau_f(dq), np.zeros(n), atol=1e-12,
        )
    text = dyn_path.read_text(encoding="utf-8")
    assert "parameters.pkl" not in text
    assert "import pickle" not in text


def test_emitted_pipeline_mode2_npz_dump(tmp_path):
    """End-to-end Mode 2 smoke with .npz trajectory dump enabled."""
    from src.pipeline import SystemIdentificationPipeline
    n = 2
    cfg = {
        "urdf_path": str(URDF_RRBOT),
        "output_dir": str(tmp_path / "out"),
        "method": "newton_euler",
        "friction": {"model": "viscous"},
        "identification": {"solver": "ols", "feasibility_method": "none"},
        "joint_limits": {
            "position": [[-3.14, 3.14]] * n,
            "velocity": [[-3.0, 3.0]] * n,
            "acceleration": [[-8.0, 8.0]] * n,
        },
        "excitation": {
            "num_harmonics": 3,
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 1,
            "optimize_condition_number": True,
            "optimizer_max_iter": 50,
        },
        "dynamics_model": {
            "export_npz": True,
            "include_coriolis_matrix": False,
            "include_friction_torque": True,
            "evaluation_points": "trajectory",
            "simplify": "trigsimp",
        },
    }
    SystemIdentificationPipeline(cfg).run()
    out = Path(cfg["output_dir"])
    assert (out / "dynamics_model.py").exists()
    assert (out / "dynamics_model.npz").exists()
    with np.load(out / "dynamics_model.npz", allow_pickle=True) as d:
        consistency = float(d["consistency_error"])
        M_traj = d["M_trajectory"]
    assert consistency < 1e-8, f"consistency_error too high: {consistency:.3e}"
    # M(q) symmetric at every sample.
    sym_err = np.max(np.abs(M_traj - M_traj.transpose(0, 2, 1)))
    assert sym_err < 1e-10, f"M symmetry violated: {sym_err:.3e}"
