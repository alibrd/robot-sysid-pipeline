"""Tests for pipeline partitioning: excitation_only / checkpoint_dir modes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

URDF_RRBOT = str(ROOT / "tests" / "assets" / "RRBot_single.urdf")


def _make_config(tmp_path, output_name="out"):
    """Minimal pipeline config for fast 2-DOF test runs."""
    return {
        "urdf_path": URDF_RRBOT,
        "output_dir": str(tmp_path / output_name),
        "method": "newton_euler",
        "joint_limits": {
            "position": [[-1, 1], [-1, 1]],
            "velocity": [[-2, 2], [-2, 2]],
            "acceleration": [[-5, 5], [-5, 5]],
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
            "feasibility_method": "none",
            "data_file": None,
        },
    }


class TestExcitationOnly:
    """Tests for excitation_only mode."""

    def test_creates_checkpoint_files(self, tmp_path):
        """excitation_only creates checkpoint.npz, checkpoint_config.json,
        and excitation_trajectory.npz."""
        from src.pipeline import SystemIdentificationPipeline

        cfg = _make_config(tmp_path)
        cfg["excitation_only"] = True
        pipeline = SystemIdentificationPipeline(cfg)
        pipeline.run()

        out = pipeline.output_dir
        assert (out / "checkpoint.npz").exists()
        assert (out / "checkpoint_config.json").exists()
        assert (out / "excitation_trajectory.npz").exists()

    def test_does_not_create_identification_results(self, tmp_path):
        """excitation_only should NOT produce identification results."""
        from src.pipeline import SystemIdentificationPipeline

        cfg = _make_config(tmp_path)
        cfg["excitation_only"] = True
        pipeline = SystemIdentificationPipeline(cfg)
        pipeline.run()

        out = pipeline.output_dir
        assert not (out / "identification_results.npz").exists()
        assert not (out / "results_summary.json").exists()

    def test_nested_pipeline_section_is_honored(self, tmp_path):
        """Workflow-style pipeline.excitation_only works in pipeline configs."""
        from src.pipeline import SystemIdentificationPipeline

        cfg = _make_config(tmp_path)
        cfg["pipeline"] = {
            "excitation_only": True,
            "checkpoint_dir": None,
        }
        pipeline = SystemIdentificationPipeline(cfg)
        pipeline.run()

        out = pipeline.output_dir
        assert (out / "checkpoint.npz").exists()
        assert (out / "excitation_trajectory.npz").exists()
        assert not (out / "identification_results.npz").exists()

    def test_checkpoint_contains_expected_keys(self, tmp_path):
        """Checkpoint npz has all required arrays."""
        from src.pipeline import SystemIdentificationPipeline

        cfg = _make_config(tmp_path)
        cfg["excitation_only"] = True
        pipeline = SystemIdentificationPipeline(cfg)
        pipeline.run()

        cp = np.load(str(pipeline.output_dir / "checkpoint.npz"),
                     allow_pickle=True)
        expected_keys = {
            "exc_params", "exc_freqs", "exc_q0", "exc_cost",
            "exc_basis", "exc_optimize_phase",
            "exc_torque_constraint_method",
            "q_data", "dq_data", "ddq_data", "tau_data", "data_fs",
            "nominal_params_used", "sequential_history",
        }
        assert expected_keys.issubset(set(cp.files)), (
            f"Missing keys: {expected_keys - set(cp.files)}"
        )


class TestFromCheckpoint:
    """Tests for checkpoint_dir (resume) mode."""

    def test_produces_identification_results(self, tmp_path):
        """Resume from checkpoint produces identification_results.npz."""
        from src.pipeline import SystemIdentificationPipeline

        # Phase 1: excitation only
        cfg1 = _make_config(tmp_path, "exc_out")
        cfg1["excitation_only"] = True
        p1 = SystemIdentificationPipeline(cfg1)
        p1.run()

        # Phase 2: resume
        cfg2 = _make_config(tmp_path, "resume_out")
        cfg2["checkpoint_dir"] = str(p1.output_dir)
        p2 = SystemIdentificationPipeline(cfg2)
        p2.run()

        assert (p2.output_dir / "identification_results.npz").exists()
        assert (p2.output_dir / "results_summary.json").exists()

    def test_partitioned_matches_full(self, tmp_path):
        """Partitioned run (exc_only + resume) produces same identified
        parameters as a full run with the same config."""
        from src.pipeline import SystemIdentificationPipeline

        # Full run
        cfg_full = _make_config(tmp_path, "full_out")
        p_full = SystemIdentificationPipeline(cfg_full)
        p_full.run()

        # Partitioned: phase 1
        cfg_exc = _make_config(tmp_path, "exc_out")
        cfg_exc["excitation_only"] = True
        p_exc = SystemIdentificationPipeline(cfg_exc)
        p_exc.run()

        # Partitioned: phase 2
        cfg_resume = _make_config(tmp_path, "resume_out")
        cfg_resume["checkpoint_dir"] = str(p_exc.output_dir)
        p_resume = SystemIdentificationPipeline(cfg_resume)
        p_resume.run()

        full = np.load(str(p_full.output_dir / "identification_results.npz"))
        resumed = np.load(str(p_resume.output_dir / "identification_results.npz"))

        np.testing.assert_allclose(
            full["pi_identified"], resumed["pi_identified"], atol=1e-10,
        )
        np.testing.assert_allclose(
            full["residual"], resumed["residual"], atol=1e-10,
        )

    def test_missing_checkpoint_raises(self, tmp_path):
        """checkpoint_dir pointing to empty dir raises FileNotFoundError."""
        from src.pipeline import SystemIdentificationPipeline

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        cfg = _make_config(tmp_path)
        cfg["checkpoint_dir"] = str(empty_dir)
        pipeline = SystemIdentificationPipeline(cfg)
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            pipeline.run()


class TestFullModeUnchanged:
    """Backward compatibility: default config (no partitioning flags)."""

    def test_full_pipeline_still_works(self, tmp_path):
        """Default run() produces all expected output files."""
        from src.pipeline import SystemIdentificationPipeline

        cfg = _make_config(tmp_path)
        pipeline = SystemIdentificationPipeline(cfg)
        pipeline.run()

        out = pipeline.output_dir
        assert (out / "identification_results.npz").exists()
        assert (out / "results_summary.json").exists()
        assert (out / "excitation_trajectory.npz").exists()
        # No checkpoint files in full mode
        assert not (out / "checkpoint.npz").exists()


class TestConfigValidation:
    """Config-level validation of partitioning fields."""

    def test_both_flags_raises(self, tmp_path):
        """excitation_only + checkpoint_dir simultaneously raises ValueError."""
        from src.config_loader import load_config_dict

        cfg = _make_config(tmp_path)
        cfg["excitation_only"] = True
        cfg["checkpoint_dir"] = str(tmp_path / "some_dir")
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_config_dict(cfg)

    def test_nested_both_flags_raises(self, tmp_path):
        """Nested pipeline mode fields are validated after normalization."""
        from src.config_loader import load_config_dict

        cfg = _make_config(tmp_path)
        cfg["pipeline"] = {
            "excitation_only": True,
            "checkpoint_dir": str(tmp_path / "some_dir"),
        }
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_config_dict(cfg)
