"""Tests for multi-run PyBullet validation benchmark export."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _write_summary(run_dir: Path,
                   robot_name: str,
                   passed: bool,
                   max_abs: float,
                   rms: float,
                   norm_rms: float):
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "robot_name": robot_name,
        "nDoF": 2,
        "joint_names": [f"{robot_name}_j1", f"{robot_name}_j2"],
        "sample_count": 200,
        "sample_rate_hz": 100.0,
        "gravity": [0.0, 0.0, -9.80665],
        "use_fixed_base": True,
        "tolerance_abs": 1e-3,
        "tolerance_normalized_rms": 1e-3,
        "method": "newton_euler",
        "max_abs_error_per_joint": [max_abs * 0.5, max_abs],
        "rms_error_per_joint": [rms * 0.5, rms],
        "normalized_rms_error_per_joint": [norm_rms * 0.5, norm_rms],
        "per_joint_pass": [passed, passed],
        "global_max_abs_error": max_abs,
        "global_rms_error": rms,
        "global_normalized_rms_error": norm_rms,
        "passed": passed,
        "notes": [],
    }
    (run_dir / "pybullet_validation_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_export_validation_benchmark_writes_csv_and_markdown(tmp_path):
    from src.pybullet_validation_benchmark import export_validation_benchmark

    _write_summary(tmp_path / "robot_b", "robot_b", False, 2e-3, 1e-3, 2e-2)
    _write_summary(tmp_path / "robot_a", "robot_a", True, 5e-4, 2e-4, 4e-3)

    result = export_validation_benchmark(str(tmp_path))

    csv_path = Path(result["benchmark_csv"])
    md_path = Path(result["benchmark_markdown"])
    assert csv_path.exists()
    assert md_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "robot_a" in csv_text
    assert "robot_b" in csv_text
    assert "Runs aggregated: `2`" in md_text
    assert "Pass count: `1`" in md_text
    assert "Fail count: `1`" in md_text
    assert md_text.index("robot_a") < md_text.index("robot_b")
