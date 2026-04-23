"""Tests for the workflow orchestration layer."""
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


def _write_excitation(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        params=np.asarray([0.05], dtype=float),
        freqs=np.asarray([0.5], dtype=float),
        q0=np.asarray([0.0], dtype=float),
        basis=np.array("cosine"),
        optimize_phase=np.array(False),
    )
    return path


def _write_pipeline_config(path: Path, urdf_path: Path, output_dir: str = "pipeline_out") -> Path:
    return _write_json(
        path,
        {
            "urdf_path": str(urdf_path),
            "output_dir": output_dir,
            "method": "newton_euler",
            "excitation": {
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 2,
            },
        },
    )


def _write_pipeline_run_config(path: Path,
                               urdf_path: Path,
                               output_dir: str = "pipeline_out",
                               friction_model: str = "none",
                               joint_limits: dict | None = None) -> Path:
    if joint_limits is None:
        joint_limits = {
            "position": [[-1.0, 1.0]],
            "velocity": [[-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0]],
        }
    return _write_json(
        path,
        {
            "urdf_path": str(urdf_path),
            "output_dir": output_dir,
            "method": "newton_euler",
            "joint_limits": joint_limits,
            "excitation": {
                "basis_functions": "cosine",
                "optimize_phase": False,
                "num_harmonics": 1,
                "base_frequency_hz": 0.5,
                "optimize_condition_number": False,
                "optimizer_max_iter": 1,
                "trajectory_duration_periods": 1,
            },
            "friction": {"model": friction_model},
            "identification": {
                "solver": "ols",
                "parameter_bounds": False,
                "feasibility_method": "none",
                "data_file": None,
            },
            "filtering": {"enabled": False},
            "downsampling": {"frequency_hz": 0},
        },
    )


def _write_validation_config(path: Path,
                             urdf_path: Path,
                             excitation_file: Path,
                             output_dir: str = "validation_out") -> Path:
    return _write_json(
        path,
        {
            "urdf_path": str(urdf_path),
            "excitation_file": str(excitation_file),
            "output_dir": output_dir,
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 2,
        },
    )


def test_pipeline_only_workflow_succeeds_without_pybullet(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_pipeline_config(tmp_path / "pipeline.json", URDF_RRBOT)
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": False,
            "run_report": False,
            "run_benchmark": False,
            "pipeline": {"config_path": str(pipeline_cfg)},
        },
    )

    calls = {}

    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg
            self.output_dir = Path(cfg["output_dir"])

        def run(self):
            calls["pipeline_ran"] = True

    monkeypatch.setattr("src.workflow.SystemIdentificationPipeline", FakePipeline)
    monkeypatch.setattr("src.workflow._is_module_available", lambda name: False)

    result = WorkflowRunner(str(workflow_cfg)).run()

    assert calls["pipeline_ran"]
    assert result["pipeline_output_dir"].endswith("pipeline_out")


def test_validation_enabled_workflow_fails_clearly_when_pybullet_missing(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    excitation_file = _write_excitation(tmp_path / "excitation.npz")
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": False,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "validation": {
                "auto_from_pipeline": False,
                "urdf_path": str(URDF_RRBOT),
                "excitation_file": str(excitation_file),
                "output_dir": "validation_out",
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 2,
            },
        },
    )

    monkeypatch.setattr(
        "src.workflow._is_module_available",
        lambda name: False if name == "pybullet" else True,
    )

    with pytest.raises(RuntimeError, match="PyBullet is required"):
        WorkflowRunner(str(workflow_cfg)).prepare()


def test_auto_from_pipeline_populates_validation_inputs(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_pipeline_config(tmp_path / "pipeline.json", URDF_RRBOT)
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "pipeline": {"config_path": str(pipeline_cfg)},
            "validation": {
                "sample_rate_hz": 120.0,
                "comparison": {
                    "tolerance_abs": 1e-3,
                    "tolerance_normalized_rms": 1e-3,
                },
            },
        },
    )

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)

    context = WorkflowRunner(str(workflow_cfg)).prepare()

    assert context["validation_cfg"]["urdf_path"] == str(URDF_RRBOT.resolve())
    assert context["validation_cfg"]["excitation_file"].endswith("pipeline_out\\excitation_trajectory.npz")
    assert context["validation_cfg"]["base_frequency_hz"] == 0.5
    assert context["validation_cfg"]["trajectory_duration_periods"] == 2
    assert context["validation_cfg"]["sample_rate_hz"] == 120.0


def test_validation_without_pipeline_missing_excitation_fails_preflight(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": False,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "validation": {
                "auto_from_pipeline": False,
                "urdf_path": str(URDF_RRBOT),
                "excitation_file": "missing_excitation.npz",
                "output_dir": "validation_out",
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 2,
            },
        },
    )

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)

    with pytest.raises(FileNotFoundError, match="Validation requires an existing excitation artifact"):
        WorkflowRunner(str(workflow_cfg)).prepare()


def test_mismatched_pipeline_and_validation_urdfs_fail_preflight(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_pipeline_config(tmp_path / "pipeline.json", URDF_RRBOT)
    excitation_file = _write_excitation(tmp_path / "excitation.npz")
    validation_cfg = _write_validation_config(
        tmp_path / "validation.json",
        URDF_PENDULUM,
        excitation_file,
    )
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "pipeline": {"config_path": str(pipeline_cfg)},
            "validation": {
                "config_path": str(validation_cfg),
                "auto_from_pipeline": False,
            },
        },
    )

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)

    with pytest.raises(ValueError, match="URDF paths differ"):
        WorkflowRunner(str(workflow_cfg)).prepare()


def test_output_root_overrides_stage_output_paths(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_pipeline_config(tmp_path / "pipeline.json", URDF_RRBOT, "custom_pipeline")
    excitation_file = _write_excitation(tmp_path / "excitation.npz")
    validation_cfg = _write_validation_config(
        tmp_path / "validation.json",
        URDF_RRBOT,
        excitation_file,
        "custom_validation",
    )
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": True,
            "output_root": "all_outputs",
            "pipeline": {"config_path": str(pipeline_cfg)},
            "validation": {
                "config_path": str(validation_cfg),
                "auto_from_pipeline": False,
                "use_external_artifacts": True,
            },
        },
    )

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)
    context = WorkflowRunner(str(workflow_cfg)).prepare()

    assert Path(context["pipeline_cfg"]["output_dir"]) == (tmp_path / "all_outputs" / "pipeline" / "pipeline")
    assert Path(context["validation_cfg"]["output_dir"]) == (tmp_path / "all_outputs" / "validation")
    assert Path(context["benchmark_validation_root"]) == (tmp_path / "all_outputs" / "validation")


def test_report_stage_uses_single_validation_dir(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    validation_dir = tmp_path / "existing_validation"
    validation_dir.mkdir()
    calls = {}

    def fake_report(path):
        calls["report_dir"] = path
        return {"report_markdown": "ok"}

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)
    monkeypatch.setattr("src.workflow.export_validation_report", fake_report)

    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": False,
            "run_validation": False,
            "run_report": True,
            "run_benchmark": False,
            "report": {"validation_dir": str(validation_dir)},
        },
    )

    WorkflowRunner(str(workflow_cfg)).run()
    assert calls["report_dir"] == str(validation_dir.resolve())


def test_benchmark_stage_uses_validation_root(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    validation_root = tmp_path / "existing_validation_root"
    validation_root.mkdir()
    calls = {}

    def fake_benchmark(root, output):
        calls["args"] = (root, output)
        return {"benchmark_markdown": "ok"}

    monkeypatch.setattr("src.workflow.export_validation_benchmark", fake_benchmark)

    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": False,
            "run_validation": False,
            "run_report": False,
            "run_benchmark": True,
            "benchmark": {"validation_root": str(validation_root)},
        },
    )

    WorkflowRunner(str(workflow_cfg)).run()
    assert calls["args"] == (str(validation_root.resolve()), str(validation_root.resolve()))


def test_workflow_relative_paths_resolve_from_workflow_file(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    config_dir = tmp_path / "configs"
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    rrbot_copy = assets_dir / "RRBot_single.urdf"
    shutil.copy2(URDF_RRBOT, rrbot_copy)
    _write_pipeline_config(config_dir / "pipeline.json", rrbot_copy, "../relative_out")
    excitation_file = _write_excitation(assets_dir / "excitation.npz")
    workflow_cfg = _write_json(
        config_dir / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "pipeline": {"config_path": "pipeline.json"},
            "validation": {
                "auto_from_pipeline": False,
                "urdf_path": "../assets/RRBot_single.urdf",
                "excitation_file": "../assets/excitation.npz",
                "output_dir": "../validation_out",
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 2,
                "use_external_artifacts": True
            },
        },
    )

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)
    context = WorkflowRunner(str(workflow_cfg)).prepare()

    assert context["pipeline_cfg"]["output_dir"] == str((config_dir / "../relative_out").resolve())
    assert context["validation_cfg"]["excitation_file"] == str(excitation_file.resolve())
    assert context["validation_cfg"]["output_dir"] == str((config_dir / "../validation_out").resolve())


def test_pipeline_config_loader_resolves_relative_paths(tmp_path):
    from src.config_loader import load_config

    config_dir = tmp_path / "cfg"
    urdf_copy = tmp_path / "assets" / "RRBot_single.urdf"
    urdf_copy.parent.mkdir(parents=True)
    shutil.copy2(URDF_RRBOT, urdf_copy)
    output_dir = "../pipeline_output"
    data_file = tmp_path / "data" / "measured.npz"
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(b"placeholder")
    config_path = _write_json(
        config_dir / "pipeline.json",
        {
            "urdf_path": "../assets/RRBot_single.urdf",
            "output_dir": output_dir,
            "identification": {"data_file": "../data/measured.npz"},
        },
    )

    cfg = load_config(str(config_path))

    assert cfg["urdf_path"] == str((config_dir / "../assets/RRBot_single.urdf").resolve())
    assert cfg["output_dir"] == str((config_dir / output_dir).resolve())
    assert cfg["identification"]["data_file"] == str((config_dir / "../data/measured.npz").resolve())


def test_validation_config_loader_resolves_relative_paths(tmp_path):
    from src.pybullet_validation import load_pybullet_validation_config

    config_dir = tmp_path / "cfg"
    urdf_copy = tmp_path / "assets" / "DrakePendulum_1DoF.urdf"
    urdf_copy.parent.mkdir(parents=True)
    shutil.copy2(URDF_PENDULUM, urdf_copy)
    excitation_file = _write_excitation(tmp_path / "data" / "excitation.npz")
    config_path = _write_json(
        config_dir / "validation.json",
        {
            "urdf_path": "../assets/DrakePendulum_1DoF.urdf",
            "excitation_file": "../data/excitation.npz",
            "output_dir": "../validation_output",
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 2,
        },
    )

    cfg = load_pybullet_validation_config(str(config_path))

    assert cfg["urdf_path"] == str((config_dir / "../assets/DrakePendulum_1DoF.urdf").resolve())
    assert cfg["excitation_file"] == str(excitation_file.resolve())
    assert cfg["output_dir"] == str((config_dir / "../validation_output").resolve())


@pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None or importlib.util.find_spec("matplotlib") is None,
    reason="pybullet and matplotlib are required for workflow integration",
)
def test_workflow_end_to_end_pipeline_validation_report_and_benchmark(tmp_path):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_pipeline_run_config(tmp_path / "pipeline.json", URDF_PENDULUM)
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": True,
            "run_benchmark": True,
            "output_root": "workflow_outputs",
            "pipeline": {"config_path": str(pipeline_cfg)},
            "validation": {
                "auto_from_pipeline": True,
                "sample_rate_hz": 100.0,
                "comparison": {
                    "tolerance_abs": 1e-4,
                    "tolerance_normalized_rms": 1e-4,
                },
            },
        },
    )

    results = WorkflowRunner(str(workflow_cfg)).run()

    pipeline_output_dir = tmp_path / "workflow_outputs" / "pipeline" / "pipeline"
    validation_output_dir = Path(results["validation_output_dir"])
    benchmark_root = tmp_path / "workflow_outputs" / "validation"

    assert results["validation_summary"]["passed"]
    assert (pipeline_output_dir / "excitation_trajectory.npz").exists()
    assert (pipeline_output_dir / "identification_results.npz").exists()
    assert (validation_output_dir / "pybullet_validation_summary.json").exists()
    assert (validation_output_dir / "pybullet_validation_data.npz").exists()
    assert (validation_output_dir / "pybullet_validation_report.md").exists()
    assert (validation_output_dir / "pybullet_validation_metrics.csv").exists()
    assert (benchmark_root / "pybullet_validation_benchmark.csv").exists()
    assert (benchmark_root / "pybullet_validation_benchmark.md").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None or importlib.util.find_spec("matplotlib") is None,
    reason="pybullet and matplotlib are required for workflow integration",
)
def test_workflow_end_to_end_rrbot_2dof(tmp_path):
    from src.workflow import WorkflowRunner

    rrbot_limits = {
        "position": [[-1.0, 1.0], [-1.0, 1.0]],
        "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
        "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
    }
    pipeline_cfg = _write_pipeline_run_config(
        tmp_path / "pipeline.json", URDF_RRBOT, joint_limits=rrbot_limits,
    )
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": True,
            "run_benchmark": True,
            "output_root": "workflow_outputs",
            "pipeline": {"config_path": str(pipeline_cfg)},
            "validation": {
                "auto_from_pipeline": True,
                "sample_rate_hz": 100.0,
                "comparison": {
                    "tolerance_abs": 1e-4,
                    "tolerance_normalized_rms": 1e-4,
                },
            },
        },
    )

    results = WorkflowRunner(str(workflow_cfg)).run()

    assert results["validation_summary"]["passed"]
    assert results["validation_summary"]["nDoF"] == 2


def test_auto_from_pipeline_friction_warning_injects_workflow_notes(tmp_path, monkeypatch):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_pipeline_run_config(
        tmp_path / "pipeline.json", URDF_RRBOT,
        friction_model="viscous",
        joint_limits={
            "position": [[-1.0, 1.0], [-1.0, 1.0]],
            "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
        },
    )
    workflow_cfg = _write_json(
        tmp_path / "workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "pipeline": {"config_path": str(pipeline_cfg)},
            "validation": {
                "auto_from_pipeline": True,
            },
        },
    )

    monkeypatch.setattr("src.workflow._is_module_available", lambda name: True)

    with pytest.warns(UserWarning, match="friction"):
        context = WorkflowRunner(str(workflow_cfg)).prepare()

    wf_notes = context["validation_cfg"].get("_workflow_notes", [])
    assert len(wf_notes) >= 1
    assert "friction" in wf_notes[0].lower()


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("pybullet") is None,
    reason="pybullet is required for elbow workflow validation",
)
def test_elbow_workflow_end_to_end(tmp_path):
    from src.workflow import WorkflowRunner

    pipeline_cfg = _write_json(
        tmp_path / "elbow_pipeline.json",
        {
            "urdf_path": str(URDF_ELBOW),
            "output_dir": str(tmp_path / "elbow_workflow" / "pipeline"),
            "method": "newton_euler",
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
            },
        },
    )
    workflow_cfg = _write_json(
        tmp_path / "elbow_workflow.json",
        {
            "run_pipeline": True,
            "run_validation": True,
            "run_report": False,
            "run_benchmark": False,
            "allow_missing_optional_dependencies": True,
            "output_root": str(tmp_path / "elbow_workflow"),
            "pipeline": {
                "config_path": str(pipeline_cfg),
            },
            "validation": {
                "auto_from_pipeline": True,
                "comparison": {
                    "tolerance_abs": 0.01,
                    "tolerance_normalized_rms": 0.2,
                },
            },
        },
    )

    results = WorkflowRunner(str(workflow_cfg)).run()

    assert (Path(results["pipeline_output_dir"]) / "identification_results.npz").exists()
    assert results["validation_summary"]["passed"] is True
