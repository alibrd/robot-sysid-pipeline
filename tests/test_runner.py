"""Tests for the unified runner orchestration layer."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_RRBOT = ASSET_DIR / "RRBot_single.urdf"
URDF_PENDULUM = ASSET_DIR / "DrakePendulum_1DoF.urdf"
URDF_ELBOW = ASSET_DIR / "ElbowManipulator_3DoF.urdf"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_excitation_artifact(path: Path,
                               freqs=None,
                               q0=None,
                               params=None) -> Path:
    """Write a minimal excitation_trajectory.npz fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if freqs is None:
        freqs = np.asarray([0.5], dtype=float)
    if q0 is None:
        q0 = np.asarray([0.0], dtype=float)
    if params is None:
        params = np.asarray([0.05], dtype=float)
    np.savez(
        str(path),
        params=np.asarray(params, dtype=float),
        freqs=np.asarray(freqs, dtype=float),
        q0=np.asarray(q0, dtype=float),
        basis=np.array("cosine"),
        optimize_phase=np.array(False),
    )
    return path


def _write_measurements_npz(path: Path,
                             n_dof: int,
                             n_samples: int = 32,
                             fs: float = 50.0,
                             tau_scale: float = 0.1) -> Path:
    """Write a minimal pipeline-compatible measurements .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, size=(n_samples, n_dof))
    dq = rng.uniform(-0.5, 0.5, size=(n_samples, n_dof))
    ddq = rng.uniform(-0.5, 0.5, size=(n_samples, n_dof))
    tau = tau_scale * rng.standard_normal(size=(n_samples, n_dof))
    np.savez(str(path), q=q, dq=dq, ddq=ddq, tau=tau, fs=fs)
    return path


def _write_identification_results_npz(path: Path,
                                      n_dof: int,
                                      friction_model: str = "none") -> Path:
    """Write a minimal identification_results.npz so MeasurementValidationRunner can load it."""
    from src.friction import friction_param_count
    path.parent.mkdir(parents=True, exist_ok=True)
    n_params = n_dof * 10 + friction_param_count(n_dof, friction_model)
    rng = np.random.default_rng(1)
    pi = rng.standard_normal(n_params) * 0.01
    np.savez(
        str(path),
        pi_corrected=pi,
        pi_identified=pi,
        method=np.array("newton_euler"),
        friction_model=np.array(friction_model),
        nDoF=np.int64(n_dof),
        feasible=np.bool_(True),
        residual=np.float64(0.0),
    )
    return path


def _make_unified_cfg(tmp_path: Path,
                      *,
                      urdf_path: Path,
                      output_dir: str | None = None,
                      stages: dict | None = None,
                      checkpoint: str | None = None,
                      excitation_overrides: dict | None = None,
                      validation_overrides: dict | None = None,
                      identification_overrides: dict | None = None,
                      friction_model: str = "none",
                      joint_limits: dict | None = None) -> Path:
    if output_dir is None:
        output_dir = str(tmp_path / "outputs")
    stages_block = {
        "excitation": True,
        "identification": False,
        "validation": False,
        "report": False,
        "benchmark": False,
        "plot": False,
    }
    if stages:
        stages_block.update(stages)
    excitation_block = {
        "basis_functions": "cosine",
        "optimize_phase": False,
        "num_harmonics": 1,
        "base_frequency_hz": 0.5,
        "optimize_condition_number": False,
        "optimizer_max_iter": 1,
        "trajectory_duration_periods": 2,
    }
    if excitation_overrides:
        excitation_block.update(excitation_overrides)
    identification_block = {
        "source": "excitation",
        "solver": "ols",
        "parameter_bounds": False,
        "feasibility_method": "none",
    }
    if identification_overrides:
        identification_block.update(identification_overrides)
    payload = {
        "urdf_path": str(urdf_path),
        "output_dir": output_dir,
        "method": "newton_euler",
        "stages": stages_block,
        "checkpoint": checkpoint,
        "excitation": excitation_block,
        "friction": {"model": friction_model},
        "identification": identification_block,
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }
    if joint_limits is not None:
        payload["joint_limits"] = joint_limits
    if validation_overrides is not None:
        payload["validation"] = validation_overrides
    return _write_json(tmp_path / "unified.json", payload)


def test_excitation_and_identification_only_succeeds_without_pybullet(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={"excitation": True, "identification": True},
    )

    calls = {}

    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg
            self.output_dir = Path(cfg["output_dir"])

        def run(self):
            calls["pipeline_ran"] = True
            calls["output_dir"] = str(self.output_dir)

    monkeypatch.setattr("src.runner.SystemIdentificationPipeline", FakePipeline)
    monkeypatch.setattr("src.runner._is_module_available", lambda name: False)

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0
    assert calls["pipeline_ran"]
    assert Path(calls["output_dir"]) == (tmp_path / "out" / "pipeline")


def test_validation_stage_fails_clearly_when_pybullet_missing(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
        },
    )

    monkeypatch.setattr(
        "src.runner._is_module_available",
        lambda name: False if name == "pybullet" else True,
    )

    with pytest.raises(RuntimeError, match="PyBullet is required"):
        UnifiedRunner(str(cfg_path)).run()


def test_validation_inputs_are_derived_from_unified_config(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
        },
        validation_overrides={
            "source": "pybullet",
            "sample_rate_hz": 120.0,
            "comparison": {
                "tolerance_abs": 1e-3,
                "tolerance_normalized_rms": 1e-3,
            },
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    runner = UnifiedRunner(str(cfg_path))
    ctx = runner._validate_and_prepare()

    assert ctx["validation_backend"] == "pybullet"
    assert ctx["validation_cfg"]["urdf_path"] == str(URDF_RRBOT.resolve())
    assert Path(ctx["validation_cfg"]["excitation_file"]) == (
        tmp_path / "out" / "pipeline" / "excitation_trajectory.npz"
    )
    assert ctx["validation_cfg"]["base_frequency_hz"] == 0.5
    assert ctx["validation_cfg"]["trajectory_duration_periods"] == 2
    assert ctx["validation_cfg"]["sample_rate_hz"] == 120.0


def test_resume_uses_checkpoint_excitation_metadata(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    # Build a fake previous run: <prev>/pipeline/{checkpoint*, excitation_trajectory*}.
    prev_run = tmp_path / "prev_run"
    pipeline_subdir = prev_run / "pipeline"
    pipeline_subdir.mkdir(parents=True)
    _write_json(
        pipeline_subdir / "checkpoint_config.json",
        {
            "excitation": {
                "base_frequency_hz": 0.1,
                "trajectory_duration_periods": 30,
            },
        },
    )
    np.savez(
        str(pipeline_subdir / "checkpoint.npz"),
        exc_freqs=np.asarray([0.1, 0.2]),
    )
    _write_excitation_artifact(
        pipeline_subdir / "excitation_trajectory.npz",
        freqs=np.asarray([0.1, 0.2]),
    )

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": False,
            "identification": True,
            "validation": True,
        },
        checkpoint=str(prev_run),
        excitation_overrides={
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 2,
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    runner = UnifiedRunner(str(cfg_path))
    ctx = runner._validate_and_prepare()

    assert ctx["validation_cfg"]["base_frequency_hz"] == 0.1
    assert ctx["validation_cfg"]["trajectory_duration_periods"] == 30


def test_resume_rejects_mismatched_checkpoint_metadata(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    prev_run = tmp_path / "prev_run"
    pipeline_subdir = prev_run / "pipeline"
    pipeline_subdir.mkdir(parents=True)
    _write_json(
        pipeline_subdir / "checkpoint_config.json",
        {
            "excitation": {
                "base_frequency_hz": 0.1,
                "trajectory_duration_periods": 30,
            },
        },
    )
    # exc_freqs that are inconsistent with base_frequency_hz=0.1 (expected
    # build_frequencies(0.1, 2) = [0.1, 0.2]).
    np.savez(
        str(pipeline_subdir / "checkpoint.npz"),
        exc_freqs=np.asarray([0.2, 0.4]),
    )
    _write_excitation_artifact(pipeline_subdir / "excitation_trajectory.npz")

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": False,
            "identification": True,
            "validation": True,
        },
        checkpoint=str(prev_run),
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="Validation excitation metadata"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_validation_without_excitation_fails_preflight(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": False,
            "identification": False,
            "validation": True,
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="validation"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_identification_only_requires_checkpoint(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": False,
            "identification": True,
            "validation": False,
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="requires 'checkpoint'"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_excitation_and_checkpoint_together_are_rejected(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": True,
            "identification": True,
        },
        checkpoint=str(tmp_path / "prev_run"),
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="'checkpoint' is incompatible"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_unified_config_rejects_pipeline_mode_keys_at_top_level(tmp_path):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": True,
            "identification": False,
            "validation": False,
        },
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["excitation_only"] = True
    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown top-level keys"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_validation_only_can_reuse_checkpoint_excitation_artifact(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    prev_run = tmp_path / "prev_run"
    pipeline_subdir = prev_run / "pipeline"
    pipeline_subdir.mkdir(parents=True)
    _write_json(
        pipeline_subdir / "checkpoint_config.json",
        {
            "excitation": {
                "base_frequency_hz": 0.1,
                "trajectory_duration_periods": 30,
            },
        },
    )
    np.savez(
        str(pipeline_subdir / "checkpoint.npz"),
        exc_freqs=np.asarray([0.1, 0.2]),
    )
    _write_excitation_artifact(
        pipeline_subdir / "excitation_trajectory.npz",
        freqs=np.asarray([0.1, 0.2]),
    )

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": False,
            "identification": False,
            "validation": True,
        },
        checkpoint=str(prev_run),
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    ctx = UnifiedRunner(str(cfg_path))._validate_and_prepare()

    assert not ctx["run_pipeline"]
    assert Path(ctx["validation_cfg"]["excitation_file"]) == (
        pipeline_subdir / "excitation_trajectory.npz"
    )
    assert ctx["validation_cfg"]["base_frequency_hz"] == 0.1
    assert ctx["validation_cfg"]["trajectory_duration_periods"] == 30


def test_output_dir_lays_out_subdirectories(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "all_outputs"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
            "benchmark": False,
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    ctx = UnifiedRunner(str(cfg_path))._validate_and_prepare()
    assert Path(ctx["pipeline_cfg"]["output_dir"]) == (
        tmp_path / "all_outputs" / "pipeline"
    )
    assert Path(ctx["validation_cfg"]["output_dir"]) == (
        tmp_path / "all_outputs" / "validation"
    )
    for runner_key in ("stages", "checkpoint", "validation", "plot", "report", "benchmark"):
        assert runner_key not in ctx["pipeline_cfg"], (
            f"Runner-only key '{runner_key}' must not appear in the pipeline config dict"
        )


def test_report_stage_uses_validation_dir(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    output_dir = tmp_path / "out"
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True)
    (validation_dir / "pybullet_validation_summary.json").write_text("{}")

    calls = {}

    def fake_report(path):
        calls["report_dir"] = path
        return {"report_markdown": "ok"}

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)
    monkeypatch.setattr("src.runner.export_validation_report", fake_report)

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(output_dir),
        stages={
            "excitation": False,
            "identification": False,
            "validation": False,
            "report": True,
        },
    )

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0
    assert calls["report_dir"] == str(validation_dir)


def test_benchmark_stage_uses_output_dir_validation(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    output_dir = tmp_path / "out"
    validation_root = output_dir / "validation"
    validation_root.mkdir(parents=True)

    calls = {}

    def fake_benchmark(root, output):
        calls["args"] = (root, output)
        return {"benchmark_markdown": "ok"}

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)
    monkeypatch.setattr("src.runner.export_validation_benchmark", fake_benchmark)

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(output_dir),
        stages={
            "excitation": False,
            "identification": False,
            "validation": False,
            "report": False,
            "benchmark": True,
        },
    )

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0
    assert calls["args"] == (str(validation_root), str(validation_root))


def test_unified_relative_paths_resolve_from_config_dir(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    config_dir = tmp_path / "configs"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    rrbot_copy = assets_dir / "RRBot_single.urdf"
    shutil.copy2(URDF_RRBOT, rrbot_copy)

    config_path = _write_json(
        config_dir / "unified.json",
        {
            "urdf_path": "../assets/RRBot_single.urdf",
            "output_dir": "../relative_out",
            "method": "newton_euler",
            "stages": {
                "excitation": True,
                "identification": True,
                "validation": False,
            },
            "checkpoint": None,
            "excitation": {
                "basis_functions": "cosine",
                "num_harmonics": 1,
                "base_frequency_hz": 0.5,
                "optimize_condition_number": False,
                "optimizer_max_iter": 1,
                "trajectory_duration_periods": 2,
            },
            "friction": {"model": "none"},
            "identification": {
                "source": "excitation",
                "solver": "ols",
                "feasibility_method": "none",
            },
        },
    )

    runner = UnifiedRunner(str(config_path))
    assert runner.cfg["urdf_path"] == str(rrbot_copy.resolve())
    assert runner.cfg["output_dir"] == str((config_dir / "../relative_out").resolve())


def test_friction_pipeline_warns_validation(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    rrbot_limits = {
        "position": [[-1.0, 1.0], [-1.0, 1.0]],
        "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
        "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
    }
    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
        },
        friction_model="viscous",
        joint_limits=rrbot_limits,
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.warns(UserWarning, match="friction"):
        ctx = UnifiedRunner(str(cfg_path))._validate_and_prepare()
    notes = ctx["validation_cfg"].get("_workflow_notes", [])
    assert len(notes) >= 1
    assert "friction" in notes[0].lower()


# ----------------------------------------------------------------------
# New source-combination tests (Mode 1 / Mode 2 / mixed)
# ----------------------------------------------------------------------


def test_id_source_path_dispatches_measurement_input(tmp_path, monkeypatch):
    """identification.source = <path> must translate to internal data_file."""
    from src.runner import UnifiedRunner

    meas_path = _write_measurements_npz(tmp_path / "data" / "measurements.npz", n_dof=2)

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": False,
            "identification": True,
            "validation": False,
        },
        identification_overrides={"source": str(meas_path)},
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    runner = UnifiedRunner(str(cfg_path))
    ctx = runner._validate_and_prepare()

    pipeline_cfg = ctx["pipeline_cfg"]
    assert pipeline_cfg["identification"]["data_file"] == str(meas_path.resolve())
    assert "source" not in pipeline_cfg["identification"]
    assert pipeline_cfg["excitation_only"] is False


def test_id_source_path_auto_disables_excitation_stage(tmp_path, monkeypatch):
    """Mode-2 identification ignores an enabled excitation stage."""
    from src.runner import UnifiedRunner

    meas_path = _write_measurements_npz(tmp_path / "data" / "measurements.npz", n_dof=2)

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": False,
        },
        identification_overrides={"source": str(meas_path)},
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    runner = UnifiedRunner(str(cfg_path))
    with pytest.warns(UserWarning, match="ignored"):
        ctx = runner._validate_and_prepare()

    assert runner.cfg["stages"]["excitation"] is False
    assert ctx["run_pipeline"] is True
    assert ctx["pipeline_cfg"]["excitation_only"] is False
    assert ctx["pipeline_cfg"]["checkpoint_dir"] is None
    assert ctx["pipeline_cfg"]["identification"]["data_file"] == str(meas_path.resolve())


def test_val_source_path_dispatches_measurement_validation(tmp_path, monkeypatch):
    """validation.source = <path> must route to MeasurementValidationRunner."""
    from src.runner import UnifiedRunner

    meas_in = _write_measurements_npz(
        tmp_path / "data" / "measurements.npz", n_dof=2
    )
    val_in = _write_measurements_npz(
        tmp_path / "val" / "measurements.npz", n_dof=2
    )
    # Pre-write identification results so the runner accepts validation-only.
    _write_identification_results_npz(
        tmp_path / "out" / "pipeline" / "identification_results.npz",
        n_dof=2,
    )

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": False,
            "identification": False,
            "validation": True,
        },
        identification_overrides={"source": str(meas_in)},
        validation_overrides={"source": str(val_in)},
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    ctx = UnifiedRunner(str(cfg_path))._validate_and_prepare()
    assert ctx["validation_backend"] == "measurements"
    assert ctx["validation_cfg"]["measurements_path"] == str(val_in.resolve())
    assert ctx["validation_cfg"]["urdf_path"] == str(URDF_RRBOT.resolve())


def test_id_path_val_pybullet_is_rejected(tmp_path, monkeypatch):
    """Mode-2 identification with PyBullet validation is explicitly unsupported."""
    from src.runner import UnifiedRunner

    meas_in = _write_measurements_npz(
        tmp_path / "data" / "measurements.npz", n_dof=2
    )

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": False,
            "identification": True,
            "validation": True,
        },
        identification_overrides={"source": str(meas_in)},
        validation_overrides={"source": "pybullet"},
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="not supported"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_id_excitation_val_path_combines_synth_id_with_real_validation(tmp_path, monkeypatch):
    """Synthetic identification + real-measurement validation must work end-to-end."""
    from src.runner import UnifiedRunner

    val_in = _write_measurements_npz(
        tmp_path / "val" / "measurements.npz", n_dof=2
    )

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "out"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
        },
        validation_overrides={"source": str(val_in)},
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    ctx = UnifiedRunner(str(cfg_path))._validate_and_prepare()
    assert ctx["validation_backend"] == "measurements"
    assert ctx["validation_cfg"]["measurements_path"] == str(val_in.resolve())
    assert ctx["pipeline_cfg"]["identification"]["data_file"] is None


def test_removed_keys_raise_helpful_error(tmp_path):
    """Old top-level keys (resume / validation_pybullet) must fail loudly."""
    from src.runner import UnifiedRunner

    for legacy_block, expected in [
        ({"resume": {"from_checkpoint": None}}, "checkpoint"),
        ({"validation_pybullet": {"sample_rate_hz": 0}}, "validation_pybullet"),
        ({"report": {}}, "report"),
        ({"benchmark": {}}, "benchmark"),
    ]:
        payload = {
            "urdf_path": str(URDF_RRBOT),
            "output_dir": str(tmp_path / "out"),
            "method": "newton_euler",
            "stages": {
                "excitation": True,
                "identification": False,
                "validation": False,
            },
            "identification": {"source": "excitation"},
            **legacy_block,
        }
        cfg_path = _write_json(tmp_path / f"legacy_{expected}.json", payload)
        with pytest.raises(ValueError, match=expected):
            UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_id_data_file_is_rejected_with_helpful_error(tmp_path):
    """identification.data_file (legacy) must point users at identification.source."""
    from src.runner import UnifiedRunner

    payload = {
        "urdf_path": str(URDF_RRBOT),
        "output_dir": str(tmp_path / "out"),
        "method": "newton_euler",
        "stages": {
            "excitation": True,
            "identification": True,
            "validation": False,
        },
        "identification": {"data_file": None, "solver": "ols", "feasibility_method": "none"},
    }
    cfg_path = _write_json(tmp_path / "legacy_data_file.json", payload)
    with pytest.raises(ValueError, match="identification.source"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


# ----------------------------------------------------------------------
# Pre-existing end-to-end tests (PyBullet smoke tests)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None
    or importlib.util.find_spec("matplotlib") is None,
    reason="pybullet and matplotlib are required for end-to-end runner integration",
)
def test_unified_end_to_end_pendulum(tmp_path):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_PENDULUM,
        output_dir=str(tmp_path / "outputs"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
            "report": True,
            "benchmark": True,
        },
        excitation_overrides={
            "basis_functions": "cosine",
            "num_harmonics": 1,
            "base_frequency_hz": 0.5,
            "optimize_condition_number": False,
            "optimizer_max_iter": 1,
            "trajectory_duration_periods": 1,
        },
        validation_overrides={
            "source": "pybullet",
            "sample_rate_hz": 100.0,
            "comparison": {
                "tolerance_abs": 1e-4,
                "tolerance_normalized_rms": 1e-4,
            },
        },
        joint_limits={
            "position": [[-1.0, 1.0]],
            "velocity": [[-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0]],
        },
    )

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0

    pipeline_output_dir = tmp_path / "outputs" / "pipeline"
    validation_root = tmp_path / "outputs" / "validation"

    assert (pipeline_output_dir / "excitation_trajectory.npz").exists()
    assert (pipeline_output_dir / "identification_results.npz").exists()

    # This whole block is skipped when pybullet is absent.
    assert (validation_root / "pybullet_validation_summary.json").exists()
    assert (validation_root / "pybullet_validation_data.npz").exists()
    assert (validation_root / "pybullet_validation_report.md").exists()
    assert (validation_root / "pybullet_validation_metrics.csv").exists()
    assert (validation_root / "pybullet_validation_benchmark.csv").exists()
    assert (validation_root / "pybullet_validation_benchmark.md").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None
    or importlib.util.find_spec("matplotlib") is None,
    reason="pybullet and matplotlib are required for end-to-end runner integration",
)
def test_unified_end_to_end_rrbot_2dof(tmp_path):
    from src.runner import UnifiedRunner

    rrbot_limits = {
        "position": [[-1.0, 1.0], [-1.0, 1.0]],
        "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
        "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
    }
    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        output_dir=str(tmp_path / "outputs"),
        stages={
            "excitation": True,
            "identification": True,
            "validation": True,
            "report": True,
            "benchmark": True,
        },
        excitation_overrides={
            "basis_functions": "cosine",
            "num_harmonics": 1,
            "base_frequency_hz": 0.5,
            "optimize_condition_number": False,
            "optimizer_max_iter": 1,
            "trajectory_duration_periods": 1,
        },
        validation_overrides={
            "source": "pybullet",
            "sample_rate_hz": 100.0,
            "comparison": {
                "tolerance_abs": 1e-4,
                "tolerance_normalized_rms": 1e-4,
            },
        },
        joint_limits=rrbot_limits,
    )

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None,
    reason="pybullet is required for elbow runner validation",
)
def test_unified_end_to_end_elbow(tmp_path):
    from src.runner import UnifiedRunner

    cfg_path = _write_json(
        tmp_path / "elbow_unified.json",
        {
            "urdf_path": str(URDF_ELBOW),
            "output_dir": str(tmp_path / "elbow_out"),
            "method": "newton_euler",
            "stages": {
                "excitation": True,
                "identification": True,
                "validation": True,
            },
            "checkpoint": None,
            "joint_limits": {
                "position": [[-3.14159, 3.14159], [-1.5708, 1.5708], [-1.5708, 1.5708]],
                "velocity": [[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]],
                "acceleration": [[-8.0, 8.0], [-8.0, 8.0], [-8.0, 8.0]],
            },
            "excitation": {
                "num_harmonics": 5,
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 10,
                "optimize_condition_number": True,
                "optimizer_max_iter": 2000,
            },
            "identification": {
                "source": "excitation",
                "solver": "ols",
                "feasibility_method": "cholesky",
            },
            "validation": {
                "source": "pybullet",
                "comparison": {
                    "tolerance_abs": 0.01,
                    "tolerance_normalized_rms": 0.2,
                },
            },
        },
    )

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0


if "--run-slow" in sys.argv:
    @pytest.mark.slow
    def test_franka_fr3_7dof_urdf_parses_and_config_validates(tmp_path):
        """FR3 config must load cleanly and produce a valid pipeline config dict."""
        from src.runner import UnifiedRunner

        config_path = Path(__file__).resolve().parent.parent / "config" / "franka_fr3_7dof.json"
        runner = UnifiedRunner(str(config_path))

        assert runner.cfg["stages"]["excitation"] is True
        assert runner.cfg["stages"]["identification"] is True
        assert runner.cfg["urdf_path"].endswith("FrankaFR3_7DoF.urdf")

        from unittest.mock import patch

        with patch("src.runner._is_module_available", return_value=True):
            ctx = runner._validate_and_prepare()
        assert ctx["run_pipeline"] is True
        assert ctx["run_validation"] is True
