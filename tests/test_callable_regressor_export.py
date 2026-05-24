# Env repair that fixed 0xc06d007f BLAS crash:
#   conda install -p C:/PyVenvs/sysid_pipeline_conda -c conda-forge --force-reinstall "libblas=*=*openblas" "libcblas=*=*openblas" "liblapack=*=*openblas"
"""Tests for the standalone callable regressor export (regressor.py + parameters.pkl)."""
import importlib.util
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URDF_RRBOT = ROOT / "tests" / "assets" / "RRBot_single.urdf"
# FingerEdu_3DoF.xacro requires the `xacro` CLI which is not available in this
# environment; use the equivalent 3-DOF URDF fixture instead.
URDF_ELBOW = ROOT / "tests" / "assets" / "ElbowManipulator_3DoF.urdf"

# ---------------------------------------------------------------------------
# Parametrization: (backend, urdf_path, urdf_id, nDoF)
# Fast fixtures: NE on both robots + EL on RRBot
# Slow fixture: EL on Elbow (symbolic 3-DOF build dominates wall time)
# ---------------------------------------------------------------------------
NE_RRBOT  = ("newton_euler",    URDF_RRBOT,  "ne_rrbot",   2)
NE_ELBOW  = ("newton_euler",    URDF_ELBOW,  "ne_elbow3",  3)
EL_RRBOT  = ("euler_lagrange",  URDF_RRBOT,  "el_rrbot",   2)
EL_ELBOW  = ("euler_lagrange",  URDF_ELBOW,  "el_elbow3",  3)

_FAST = [NE_RRBOT, NE_ELBOW, EL_RRBOT]
_FAST_IDS = [x[2] for x in _FAST]

_SLOW = [EL_ELBOW]
_SLOW_IDS = [x[2] for x in _SLOW]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_cfg(tmp_path, *, urdf, excitation_only, friction_model, method,
                 nDoF, feasibility_method="none"):
    pos_lim = [[-3.14159, 3.14159]] * nDoF
    vel_lim = [[-3.0, 3.0]] * nDoF
    acc_lim = [[-8.0, 8.0]] * nDoF
    return {
        "urdf_path": str(urdf),
        "output_dir": str(tmp_path / "out"),
        "method": method,
        "excitation_only": excitation_only,
        "friction": {"model": friction_model},
        "identification": {
            "solver": "ols",
            "feasibility_method": feasibility_method,
        },
        "joint_limits": {
            "position": pos_lim,
            "velocity": vel_lim,
            "acceleration": acc_lim,
        },
        "excitation": {
            "num_harmonics": 3,
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 1,
            "optimize_condition_number": True,
            "optimizer_max_iter": 50,
        },
    }


def _load_emitted_module(path: Path):
    spec = importlib.util.spec_from_file_location("_emitted_regressor", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_state(nDoF: int, seed: int):
    rng = np.random.default_rng(seed)
    q   = rng.uniform(-0.5, 0.5, nDoF)
    dq  = rng.uniform(-0.5, 0.5, nDoF)
    ddq = rng.uniform(-1.0, 1.0, nDoF)
    return q, dq, ddq


def _run_excitation_only(tmp_path, *, urdf, method, friction_model, nDoF):
    from src.pipeline import SystemIdentificationPipeline

    cfg = _minimal_cfg(
        tmp_path, urdf=urdf, excitation_only=True,
        friction_model=friction_model, method=method, nDoF=nDoF,
    )
    SystemIdentificationPipeline(cfg).run()
    return Path(cfg["output_dir"])


def _run_full_pipeline(tmp_path, *, urdf, method, friction_model, nDoF,
                       feasibility_method="none"):
    from src.pipeline import SystemIdentificationPipeline

    cfg = _minimal_cfg(
        tmp_path, urdf=urdf, excitation_only=False,
        friction_model=friction_model, method=method, nDoF=nDoF,
        feasibility_method=feasibility_method,
    )
    SystemIdentificationPipeline(cfg).run()
    return Path(cfg["output_dir"])


# ---------------------------------------------------------------------------
# Test 1 – excitation-only emits regressor.py, parameters.pkl, checkpoint.npz,
#          excitation_trajectory.npz; pickle has kind=="nominal".
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_excitation_only_emits_external_artifacts(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)

    assert (out / "regressor.py").exists(), "regressor.py not emitted"
    assert (out / "parameters.pkl").exists(), "parameters.pkl not emitted"
    assert (out / "checkpoint.npz").exists(), "checkpoint.npz not emitted"
    assert (out / "excitation_trajectory.npz").exists(), \
        "excitation_trajectory.npz not emitted"

    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)

    assert pkl["kind"] == "nominal"
    assert pkl["residual"] is None
    assert pkl["feasibility_method"] is None

    n = pkl["nDoF"]
    n_fric = pkl["n_friction_params"]
    assert pkl["pi_rigid"].size == 10 * n
    assert pkl["pi_friction"].size == n_fric
    np.testing.assert_array_equal(
        np.concatenate([pkl["pi_rigid"], pkl["pi_friction"]]),
        pkl["pi_augmented"],
    )


@pytest.mark.slow
@pytest.mark.parametrize("backend,urdf,_id,nDoF", _SLOW, ids=_SLOW_IDS)
def test_excitation_only_emits_external_artifacts_slow(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)
    assert (out / "regressor.py").exists()
    assert (out / "parameters.pkl").exists()
    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)
    assert pkl["kind"] == "nominal"


# ---------------------------------------------------------------------------
# Test 2 – full pipeline emits artifacts; kind=="identified", residual is float.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_full_pipeline_emits_external_artifacts(tmp_path, backend, urdf, _id, nDoF):
    out = _run_full_pipeline(tmp_path, urdf=urdf, method=backend,
                             friction_model="viscous", nDoF=nDoF)

    assert (out / "regressor.py").exists()
    assert (out / "parameters.pkl").exists()

    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)

    assert pkl["kind"] == "identified"
    assert isinstance(pkl["residual"], float)
    assert pkl["feasibility_method"] is not None

    required_keys = {
        "pi", "pi_augmented", "pi_rigid", "pi_friction", "kind",
        "nDoF", "joint_names", "link_names", "friction_model", "backend",
        "rigid_parameter_names", "friction_parameter_names",
        "augmented_parameter_names", "n_rigid_params", "n_friction_params",
        "n_augmented_params", "gravity", "residual", "feasibility_method",
    }
    assert required_keys <= pkl.keys(), f"Missing keys: {required_keys - pkl.keys()}"


@pytest.mark.slow
@pytest.mark.parametrize("backend,urdf,_id,nDoF", _SLOW, ids=_SLOW_IDS)
def test_full_pipeline_emits_external_artifacts_slow(tmp_path, backend, urdf, _id, nDoF):
    out = _run_full_pipeline(tmp_path, urdf=urdf, method=backend,
                             friction_model="viscous", nDoF=nDoF)
    assert (out / "parameters.pkl").exists()
    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)
    assert pkl["kind"] == "identified"


# ---------------------------------------------------------------------------
# Test 3 – emitted regressor.py has no src.* / sysid_pipeline imports.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_emitted_regressor_is_numpy_only(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="none", nDoF=nDoF)
    text = (out / "regressor.py").read_text(encoding="utf-8")

    for banned in ("from src", "import src", "import sysid", "from sysid_pipeline"):
        assert banned not in text, f"Banned import found: {banned!r}"

    import_lines = [ln.strip() for ln in text.splitlines()
                    if re.match(r"^\s*(import|from)\s+", ln)]
    allowed_top = {"numpy", "pickle", "pathlib", "__future__"}
    for line in import_lines:
        m = re.match(r"(?:import|from)\s+(\w+)", line)
        if m:
            top = m.group(1)
            assert top in allowed_top, \
                f"Unexpected top-level import {top!r} in emitted regressor"


# ---------------------------------------------------------------------------
# Test 4 – emitted matrix values match in-memory RegressorModel.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("friction_model", ["none", "viscous_coulomb"])
@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_emitted_regressor_matches_in_memory_model(
        tmp_path, backend, urdf, _id, nDoF, friction_model):
    from src.regressor_model import RegressorModel

    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model=friction_model, nDoF=nDoF)
    mod = _load_emitted_module(out / "regressor.py")

    model = RegressorModel.from_urdf(
        urdf, friction_model=friction_model, backend=backend,
        cache_dir=tmp_path / "el_cache",
    )
    n = model.nDoF
    n_aug = model.n_augmented_params

    for seed in range(5):
        q, dq, ddq = _random_state(n, seed)
        np.testing.assert_allclose(
            mod.Y_rigid(q, dq, ddq), model.rigid(q, dq, ddq),
            atol=1e-10, err_msg=f"Y_rigid mismatch seed={seed}",
        )
        np.testing.assert_allclose(
            mod.Y_augmented(q, dq, ddq), model.augmented(q, dq, ddq),
            atol=1e-10, err_msg=f"Y_augmented mismatch seed={seed}",
        )
        np.testing.assert_allclose(
            mod.Y(q, dq, ddq), model.augmented(q, dq, ddq),
            atol=1e-10, err_msg=f"Y alias mismatch seed={seed}",
        )

    assert mod.Y_rigid(q, dq, ddq).shape == (n, 10 * n)
    assert mod.Y_augmented(q, dq, ddq).shape == (n, n_aug)


# ---------------------------------------------------------------------------
# Test 5 – tau with no pi arg auto-loads sibling pickle.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_tau_auto_loads_sibling_pickle(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)
    mod = _load_emitted_module(out / "regressor.py")

    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)
    pi_aug = np.asarray(pkl["pi_augmented"], dtype=float)

    q, dq, ddq = _random_state(nDoF, seed=99)
    np.testing.assert_allclose(
        mod.tau(q, dq, ddq),
        mod.Y_augmented(q, dq, ddq) @ pi_aug,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Test 6 – tau rejects a bad-length pi with ValueError mentioning lengths.
# ---------------------------------------------------------------------------

def test_tau_rejects_bad_length_pi(tmp_path):
    out = _run_excitation_only(tmp_path, urdf=URDF_RRBOT,
                               method="newton_euler",
                               friction_model="viscous", nDoF=2)
    mod = _load_emitted_module(out / "regressor.py")
    q, dq, ddq = _random_state(2, seed=42)

    with pytest.raises(ValueError, match=r"\d+"):
        mod.tau(q, dq, ddq, pi=np.zeros(3))


# ---------------------------------------------------------------------------
# Test 7 – tau dispatches on pi size (rigid vs augmented).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_tau_dispatches_on_pi_size(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)
    mod = _load_emitted_module(out / "regressor.py")

    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)
    pi_rigid = np.asarray(pkl["pi_rigid"], dtype=float)
    pi_aug   = np.asarray(pkl["pi_augmented"], dtype=float)

    q, dq, ddq = _random_state(nDoF, seed=7)
    np.testing.assert_allclose(
        mod.tau(q, dq, ddq, pi=pi_rigid),
        mod.Y_rigid(q, dq, ddq) @ pi_rigid,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mod.tau(q, dq, ddq, pi=pi_aug),
        mod.Y_augmented(q, dq, ddq) @ pi_aug,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Test 8 – Y_stack accepts (N, nDoF) and (nDoF, N) inputs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_y_stack_accepts_both_batch_shapes(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)
    mod = _load_emitted_module(out / "regressor.py")

    n_aug = mod.META["n_augmented_params"]
    N = 4
    rng = np.random.default_rng(11)
    q_traj   = rng.uniform(-0.5, 0.5, (N, nDoF))
    dq_traj  = rng.uniform(-0.5, 0.5, (N, nDoF))
    ddq_traj = rng.uniform(-1.0, 1.0, (N, nDoF))

    W_row = mod.Y_stack(q_traj, dq_traj, ddq_traj)
    assert W_row.shape == (N * nDoF, n_aug), \
        f"Expected ({N * nDoF}, {n_aug}), got {W_row.shape}"

    W_col = mod.Y_stack(q_traj.T, dq_traj.T, ddq_traj.T)
    assert np.array_equal(W_row, W_col), \
        "Y_stack(N,nDoF) and Y_stack(nDoF,N) gave different results"


# ---------------------------------------------------------------------------
# Test 9 – tau_traj accepts (N, nDoF) and (nDoF, N) inputs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_tau_traj_accepts_both_batch_shapes(tmp_path, backend, urdf, _id, nDoF):
    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)
    mod = _load_emitted_module(out / "regressor.py")

    with open(out / "parameters.pkl", "rb") as f:
        pkl = pickle.load(f)
    pi_aug = np.asarray(pkl["pi_augmented"], dtype=float)

    N = 4
    rng = np.random.default_rng(22)
    q_traj   = rng.uniform(-0.5, 0.5, (N, nDoF))
    dq_traj  = rng.uniform(-0.5, 0.5, (N, nDoF))
    ddq_traj = rng.uniform(-1.0, 1.0, (N, nDoF))

    T_row = mod.tau_traj(q_traj, dq_traj, ddq_traj, pi=pi_aug)
    assert T_row.shape == (N, nDoF)

    T_col = mod.tau_traj(q_traj.T, dq_traj.T, ddq_traj.T, pi=pi_aug)
    assert np.array_equal(T_row, T_col), \
        "tau_traj(N,nDoF) and tau_traj(nDoF,N) gave different results"


# ---------------------------------------------------------------------------
# Test 10 – META dict mirrors RegressorModel methods.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend,urdf,_id,nDoF", _FAST, ids=_FAST_IDS)
def test_meta_dict_mirrors_model(tmp_path, backend, urdf, _id, nDoF):
    from src.regressor_model import RegressorModel

    out = _run_excitation_only(tmp_path, urdf=urdf, method=backend,
                               friction_model="viscous", nDoF=nDoF)
    mod = _load_emitted_module(out / "regressor.py")

    model = RegressorModel.from_urdf(
        urdf, friction_model="viscous", backend=backend,
        cache_dir=tmp_path / "el_cache",
    )

    assert mod.META["backend"] == model.backend
    assert mod.META["nDoF"] == model.nDoF
    assert mod.META["joint_names"] == model.joint_names()
    assert mod.META["link_names"] == model.link_names()
    assert mod.META["rigid_parameter_names"] == model.rigid_parameter_names()
    assert mod.META["augmented_parameter_names"] == model.augmented_parameter_names()
    assert mod.META["friction_model"] == model.friction_model
    np.testing.assert_allclose(
        mod.META["gravity"],
        model.metadata_dict()["gravity"],
        atol=1e-15,
    )
