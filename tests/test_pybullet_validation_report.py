"""Tests for the PyBullet validation report exporter."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _write_summary(run_dir: Path, nDoF: int = 2):
    """Write a minimal but complete validation summary JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "robot_name": "test_robot",
        "nDoF": nDoF,
        "joint_names": [f"joint_{i}" for i in range(nDoF)],
        "sample_count": 50,
        "sample_rate_hz": 100.0,
        "gravity": [0.0, 0.0, -9.80665],
        "use_fixed_base": True,
        "tolerance_abs": 1e-3,
        "tolerance_normalized_rms": 1e-3,
        "method": "newton_euler",
        "max_abs_error_per_joint": [1e-5] * nDoF,
        "rms_error_per_joint": [5e-6] * nDoF,
        "normalized_rms_error_per_joint": [2e-4] * nDoF,
        "per_joint_pass": [True] * nDoF,
        "global_max_abs_error": 1e-5,
        "global_rms_error": 5e-6,
        "global_normalized_rms_error": 2e-4,
        "passed": True,
        "notes": ["Test note"],
    }
    (run_dir / "pybullet_validation_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return summary


def _write_data(run_dir: Path, nDoF: int = 2, N: int = 50):
    """Write a minimal validation data NPZ."""
    run_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 0.5, N)
    tau = np.random.default_rng(42).uniform(-1, 1, (N, nDoF))
    np.savez(
        str(run_dir / "pybullet_validation_data.npz"),
        t=t,
        tau_pipeline=tau,
        tau_pybullet=tau + 1e-6,
        tau_abs_error=np.full((N, nDoF), 1e-6),
    )


def test_load_validation_summary_reads_canonical_keys(tmp_path):
    from src.pybullet_validation_report import load_validation_summary

    _write_summary(tmp_path)
    summary = load_validation_summary(str(tmp_path))
    assert summary["robot_name"] == "test_robot"
    assert summary["tolerance_normalized_rms"] == 1e-3


def test_load_validation_summary_migrates_deprecated_tolerance_rel(tmp_path):
    from src.pybullet_validation_report import load_validation_summary

    tmp_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "robot_name": "legacy_robot",
        "tolerance_rel": 0.005,
    }
    (tmp_path / "pybullet_validation_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    loaded = load_validation_summary(str(tmp_path))
    assert loaded["tolerance_normalized_rms"] == 0.005


def test_load_validation_summary_raises_on_missing_file(tmp_path):
    from src.pybullet_validation_report import load_validation_summary

    with pytest.raises(FileNotFoundError):
        load_validation_summary(str(tmp_path))


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="matplotlib is not installed")
def test_export_validation_report_writes_csv_and_markdown(tmp_path):
    from src.pybullet_validation_report import export_validation_report

    _write_summary(tmp_path)
    _write_data(tmp_path)

    result = export_validation_report(str(tmp_path))

    csv_path = Path(result["metrics_csv"])
    md_path = Path(result["report_markdown"])
    assert csv_path.exists()
    assert md_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "joint_0" in csv_text
    assert "joint_1" in csv_text

    md_text = md_path.read_text(encoding="utf-8")
    assert "test_robot" in md_text
    assert "PASS" in md_text

    assert len(result["plot_paths"]) > 0
    for plot_path in result["plot_paths"]:
        assert Path(plot_path).exists()
