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


def _make_unified_cfg(tmp_path: Path,
                      *,
                      urdf_path: Path,
                      output_dir: str | None = None,
                      stages: dict | None = None,
                      resume_from: str | None = None,
                      excitation_overrides: dict | None = None,
                      validation_overrides: dict | None = None,
                      friction_model: str = "none",
                      joint_limits: dict | None = None) -> Path:
    if output_dir is None:
        output_dir = str(tmp_path / "outputs")
    stages_block = {
        "excitation": True,
        "identification": False,
        "validation_pybullet": False,
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
    payload = {
        "urdf_path": str(urdf_path),
        "output_dir": output_dir,
        "method": "newton_euler",
        "stages": stages_block,
        "resume": {"from_checkpoint": resume_from},
        "excitation": excitation_block,
        "friction": {"model": friction_model},
        "identification": {
            "solver": "ols",
            "parameter_bounds": False,
            "feasibility_method": "none",
            "data_file": None,
        },
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }
    if joint_limits is not None:
        payload["joint_limits"] = joint_limits
    if validation_overrides is not None:
        payload["validation_pybullet"] = validation_overrides
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
            "validation_pybullet": True,
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
            "validation_pybullet": True,
        },
        validation_overrides={
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
            "validation_pybullet": True,
        },
        resume_from=str(prev_run),
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
            "validation_pybullet": True,
        },
        resume_from=str(prev_run),
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
            "validation_pybullet": True,
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="validation_pybullet"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_identification_only_requires_resume_checkpoint(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": False,
            "identification": True,
            "validation_pybullet": False,
        },
    )

    monkeypatch.setattr("src.runner._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="requires resume.from_checkpoint"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_unified_config_rejects_pipeline_mode_keys_at_top_level(tmp_path):
    from src.runner import UnifiedRunner

    cfg_path = _make_unified_cfg(
        tmp_path,
        urdf_path=URDF_RRBOT,
        stages={
            "excitation": True,
            "identification": False,
            "validation_pybullet": False,
        },
    )
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    payload["excitation_only"] = True
    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown top-level keys"):
        UnifiedRunner(str(cfg_path))._validate_and_prepare()


def test_validation_only_can_reuse_resume_excitation_artifact(tmp_path, monkeypatch):
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
            "validation_pybullet": True,
        },
        resume_from=str(prev_run),
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
            "validation_pybullet": True,
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


def test_report_stage_uses_validation_dir(tmp_path, monkeypatch):
    from src.runner import UnifiedRunner

    output_dir = tmp_path / "out"
    validation_dir = output_dir / "validation" / "rrbot_single"
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
            "validation_pybullet": False,
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
            "validation_pybullet": False,
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
                "validation_pybullet": False,
            },
            "resume": {"from_checkpoint": None},
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
                "solver": "ols",
                "feasibility_method": "none",
                "data_file": None,
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
            "validation_pybullet": True,
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
            "validation_pybullet": True,
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

    # Find the validation subdirectory by robot name.
    candidates = [c for c in validation_root.iterdir() if c.is_dir()]
    assert candidates, "Validation should produce at least one robot subdirectory"
    val_dir = candidates[0]
    assert (val_dir / "pybullet_validation_summary.json").exists()
    assert (val_dir / "pybullet_validation_data.npz").exists()
    assert (val_dir / "pybullet_validation_report.md").exists()
    assert (val_dir / "pybullet_validation_metrics.csv").exists()
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
            "validation_pybullet": True,
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
                "validation_pybullet": True,
            },
            "resume": {"from_checkpoint": None},
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
                "solver": "ols",
                "feasibility_method": "cholesky",
                "data_file": None,
            },
            "validation_pybullet": {
                "comparison": {
                    "tolerance_abs": 0.01,
                    "tolerance_normalized_rms": 0.2,
                },
            },
        },
    )

    rc = UnifiedRunner(str(cfg_path)).run()
    assert rc == 0
