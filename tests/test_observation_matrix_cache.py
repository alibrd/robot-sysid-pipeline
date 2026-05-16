"""Tests for the observation-matrix cache artifact."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URDF_RRBOT = ROOT / "tests" / "assets" / "RRBot_single.urdf"


def _write_data(path: Path, *, tau_offset: float = 0.0) -> Path:
    rng = np.random.default_rng(123)
    n_samples = 40
    q = rng.uniform(-0.5, 0.5, size=(n_samples, 2))
    dq = rng.uniform(-0.8, 0.8, size=(n_samples, 2))
    ddq = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    tau = rng.normal(size=(n_samples, 2)) + tau_offset
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), q=q, dq=dq, ddq=ddq, tau=tau, fs=np.float64(100.0))
    return path


def _make_cfg(tmp_path: Path,
              output_name: str,
              data_file: Path,
              *,
              load_from=None,
              force_load: bool = False) -> dict:
    return {
        "urdf_path": str(URDF_RRBOT),
        "output_dir": str(tmp_path / output_name),
        "method": "newton_euler",
        "joint_limits": {
            "position": [[-1.0, 1.0], [-1.0, 1.0]],
            "velocity": [[-2.0, 2.0], [-2.0, 2.0]],
            "acceleration": [[-5.0, 5.0], [-5.0, 5.0]],
        },
        "excitation": {
            "basis_functions": "cosine",
            "optimize_phase": False,
            "num_harmonics": 1,
            "base_frequency_hz": 0.5,
            "optimize_condition_number": False,
            "optimizer_max_iter": 1,
            "trajectory_duration_periods": 1,
        },
        "friction": {"model": "none"},
        "identification": {
            "solver": "ols",
            "parameter_bounds": False,
            "feasibility_method": "none",
            "data_file": str(data_file),
            "observation_matrix_cache": {
                "save": True,
                "filename": "observation_matrix_cache.npz",
                "load_from": None if load_from is None else str(load_from),
                "force_load": force_load,
            },
        },
        "filtering": {"enabled": False},
        "downsampling": {"frequency_hz": 0},
    }


def test_fresh_run_writes_results_and_observation_matrix_cache(tmp_path):
    from src.pipeline import SystemIdentificationPipeline

    data_file = _write_data(tmp_path / "data" / "measured.npz")
    pipe = SystemIdentificationPipeline(_make_cfg(tmp_path, "out", data_file))
    pipe.run()

    out = pipe.output_dir
    assert (out / "identification_results.npz").exists()
    assert (out / "observation_matrix_cache.npz").exists()

    cache = np.load(str(out / "observation_matrix_cache.npz"), allow_pickle=True)
    assert {"W", "W_base", "P_matrix", "kept_cols", "rank"}.issubset(cache.files)
    assert "pi_identified" not in cache.files
    assert "pi_base" not in cache.files
    assert "pi_corrected" not in cache.files

    summary = json.loads((out / "results_summary.json").read_text())
    assert summary["observation_matrix_cache"]["status"] == "computed"
    assert summary["observation_matrix_cache"]["path"].endswith(
        "observation_matrix_cache.npz"
    )


def test_cache_reuse_skips_regressor_and_base_reduction(tmp_path, monkeypatch):
    from src.pipeline import SystemIdentificationPipeline

    data_file = _write_data(tmp_path / "data" / "measured.npz")
    first = SystemIdentificationPipeline(_make_cfg(tmp_path, "first", data_file))
    first.run()

    def fail_build(*args, **kwargs):
        raise AssertionError("build_observation_matrix should not be called")

    def fail_base(*args, **kwargs):
        raise AssertionError("compute_base_parameters should not be called")

    monkeypatch.setattr("src.pipeline.build_observation_matrix", fail_build)
    monkeypatch.setattr("src.pipeline.compute_base_parameters", fail_base)

    second = SystemIdentificationPipeline(
        _make_cfg(tmp_path, "second", data_file, load_from=first.output_dir)
    )
    second.run()

    first_results = np.load(str(first.output_dir / "identification_results.npz"),
                            allow_pickle=True)
    second_results = np.load(str(second.output_dir / "identification_results.npz"),
                             allow_pickle=True)
    np.testing.assert_allclose(
        first_results["pi_identified"],
        second_results["pi_identified"],
        atol=1e-10,
    )

    summary = json.loads((second.output_dir / "results_summary.json").read_text())
    assert summary["observation_matrix_cache"]["status"] == "loaded"
    assert summary["observation_matrix_cache"]["source_path"].endswith(
        "observation_matrix_cache.npz"
    )


def test_cache_reuse_rejects_tau_mismatch_unless_forced(tmp_path, monkeypatch):
    from src.pipeline import SystemIdentificationPipeline

    data_file = _write_data(tmp_path / "data" / "measured.npz")
    changed_tau_file = _write_data(
        tmp_path / "data" / "changed_tau.npz", tau_offset=10.0
    )
    first = SystemIdentificationPipeline(_make_cfg(tmp_path, "first", data_file))
    first.run()

    strict = SystemIdentificationPipeline(
        _make_cfg(tmp_path, "strict", changed_tau_file, load_from=first.output_dir)
    )
    with pytest.raises(ValueError, match="metadata does not match"):
        strict.run()

    def fail_build(*args, **kwargs):
        raise AssertionError("build_observation_matrix should not be called")

    def fail_base(*args, **kwargs):
        raise AssertionError("compute_base_parameters should not be called")

    monkeypatch.setattr("src.pipeline.build_observation_matrix", fail_build)
    monkeypatch.setattr("src.pipeline.compute_base_parameters", fail_base)

    forced = SystemIdentificationPipeline(
        _make_cfg(
            tmp_path,
            "forced",
            changed_tau_file,
            load_from=first.output_dir,
            force_load=True,
        )
    )
    forced.run()

    first_results = np.load(str(first.output_dir / "identification_results.npz"),
                            allow_pickle=True)
    forced_results = np.load(str(forced.output_dir / "identification_results.npz"),
                             allow_pickle=True)
    assert not np.allclose(
        first_results["pi_base"], forced_results["pi_base"], atol=1e-8
    )

    summary = json.loads((forced.output_dir / "results_summary.json").read_text())
    cache_summary = summary["observation_matrix_cache"]
    assert cache_summary["status"] == "force_loaded"
    assert cache_summary["mismatches"]
    assert any("tau" in item for item in cache_summary["mismatches"])


def test_cache_load_from_resolves_relative_paths(tmp_path):
    from src.config_loader import load_config
    from src.runner import UnifiedRunner

    config_dir = tmp_path / "cfg"
    cache_dir = tmp_path / "previous" / "pipeline"
    cache_dir.mkdir(parents=True)
    cache_ref = "../previous"
    payload = {
        "urdf_path": str(URDF_RRBOT),
        "output_dir": "../out",
        "identification": {
            "data_file": None,
            "observation_matrix_cache": {
                "load_from": cache_ref,
            },
        },
    }
    config_path = config_dir / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_config(str(config_path))
    expected = str((config_dir / cache_ref).resolve())
    assert cfg["identification"]["observation_matrix_cache"]["load_from"] == expected

    runner = UnifiedRunner(str(config_path))
    assert (
        runner.cfg["identification"]["observation_matrix_cache"]["load_from"]
        == expected
    )
