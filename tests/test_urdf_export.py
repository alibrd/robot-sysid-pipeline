"""Tests for the adapted-URDF export (Stage 12 of Mode 2).

Exercises both the importable function in :mod:`src.urdf_exporter` and the
opt-in pipeline hook in :class:`src.pipeline.SystemIdentificationPipeline`.
The PyBullet round-trip is gated behind ``--run-slow`` because it requires
the optional ``pybullet`` dependency and is materially slower than the
parser-based tests.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_RRBOT = str(ASSET_DIR / "RRBot_single.urdf")
URDF_ELBOW = str(ASSET_DIR / "ElbowManipulator_3DoF.urdf")
URDF_FR3 = str(ASSET_DIR / "FrankaFR3_7DoF.urdf")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Direct exporter — round-trip through parse_urdf
# ──────────────────────────────────────────────────────────────────────────────


class TestExportRoundTrip:
    """The adapted URDF must re-parse to the exact same parameter vector."""

    def test_rrbot_no_friction_round_trip(self, tmp_path):
        from src.kinematics import RobotKinematics
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_RRBOT)
        kin = RobotKinematics(robot)
        pi_full = kin.PI.flatten()
        assert pi_full.size == 10 * robot.nDoF

        out_urdf = tmp_path / "adapted_rrbot.urdf"
        meta = export_adapted_urdf(
            input_urdf_path=URDF_RRBOT,
            pi_full=pi_full,
            n_dof=robot.nDoF,
            friction_model="none",
            output_urdf_path=out_urdf,
            friction_sidecar_path=None,
        )
        assert Path(meta["adapted_urdf_path"]) == out_urdf
        assert meta["friction_sidecar_path"] is None
        assert meta["n_friction_params"] == 0

        robot2 = parse_urdf(str(out_urdf))
        kin2 = RobotKinematics(robot2)
        assert robot2.nDoF == robot.nDoF
        assert robot2.revolute_joint_names == robot.revolute_joint_names
        np.testing.assert_allclose(kin2.PI.flatten(), pi_full, atol=1e-9, rtol=1e-9)

    def test_elbow_round_trip_preserves_chain_order(self, tmp_path):
        from src.kinematics import RobotKinematics
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_ELBOW)
        kin = RobotKinematics(robot)
        pi_full = kin.PI.flatten()

        out_urdf = tmp_path / "adapted_elbow.urdf"
        meta = export_adapted_urdf(
            input_urdf_path=URDF_ELBOW,
            pi_full=pi_full,
            n_dof=robot.nDoF,
            friction_model="none",
            output_urdf_path=out_urdf,
        )
        assert meta["revolute_joint_names"] == ["joint1", "joint2", "joint3"]

        robot2 = parse_urdf(str(out_urdf))
        kin2 = RobotKinematics(robot2)
        np.testing.assert_allclose(kin2.PI.flatten(), pi_full, atol=1e-9, rtol=1e-9)

    def test_perturbed_inertials_round_trip(self, tmp_path):
        """A perturbed parameter vector must also round-trip exactly."""
        from src.kinematics import RobotKinematics
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_FR3)
        kin = RobotKinematics(robot)
        pi_full = kin.PI.flatten().copy()
        rng = np.random.default_rng(seed=42)
        # Perturb masses by ±5 % and COMs by a few mm.  Inertia entries are
        # left untouched so the perturbed state still corresponds to a
        # plausible (positive-definite) inertia tensor at the origin.
        for j in range(robot.nDoF):
            base = 10 * j
            pi_full[base] *= 1.0 + 0.05 * rng.standard_normal()
            pi_full[base + 1 : base + 4] += 1e-3 * rng.standard_normal(3) * pi_full[base]

        out_urdf = tmp_path / "perturbed_fr3.urdf"
        export_adapted_urdf(
            input_urdf_path=URDF_FR3,
            pi_full=pi_full,
            n_dof=robot.nDoF,
            friction_model="none",
            output_urdf_path=out_urdf,
        )
        kin2 = RobotKinematics(parse_urdf(str(out_urdf)))
        np.testing.assert_allclose(kin2.PI.flatten(), pi_full, atol=1e-7, rtol=1e-7)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Friction sidecar shape and contents
# ──────────────────────────────────────────────────────────────────────────────


class TestFrictionSidecar:
    """The JSON sidecar must carry the full identified asymmetric model."""

    def _make_synthetic_friction_pi(self, n_dof: int, model: str):
        """Return (pi_full, theta_f) for the FR3 URDF + a synthetic friction tail."""
        from src.kinematics import RobotKinematics
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_FR3)
        assert robot.nDoF == n_dof
        rigid = RobotKinematics(robot).PI.flatten()

        rng = np.random.default_rng(seed=7)
        if model == "viscous":
            theta_f = 0.1 + 0.05 * rng.standard_normal(n_dof)
        elif model == "coulomb":
            theta_f = np.concatenate([
                0.20 + 0.05 * rng.standard_normal(n_dof),
                0.18 + 0.05 * rng.standard_normal(n_dof),
            ])
        elif model == "viscous_coulomb":
            theta_f = np.concatenate([
                0.10 + 0.05 * rng.standard_normal(n_dof),
                0.20 + 0.05 * rng.standard_normal(n_dof),
                0.18 + 0.05 * rng.standard_normal(n_dof),
            ])
        else:
            theta_f = np.zeros(0)
        return np.concatenate([rigid, theta_f]), theta_f

    def test_viscous_coulomb_sidecar_contents(self, tmp_path):
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        n_dof = 7
        pi_full, theta_f = self._make_synthetic_friction_pi(n_dof, "viscous_coulomb")
        out_urdf = tmp_path / "adapted_fr3.urdf"
        out_json = tmp_path / "adapted_friction.json"
        export_adapted_urdf(
            input_urdf_path=URDF_FR3,
            pi_full=pi_full,
            n_dof=n_dof,
            friction_model="viscous_coulomb",
            output_urdf_path=out_urdf,
            friction_sidecar_path=out_json,
        )
        sidecar = json.loads(out_json.read_text())
        assert sidecar["friction_model"] == "viscous_coulomb"
        joints = sidecar["joints"]
        assert len(joints) == n_dof

        expected_names = parse_urdf(URDF_FR3).revolute_joint_names
        for j_idx, entry in enumerate(joints):
            assert entry["name"] == expected_names[j_idx]
            assert entry["Fv_viscous"] == pytest.approx(theta_f[j_idx])
            assert entry["Fcp_coulomb_positive"] == pytest.approx(theta_f[n_dof + j_idx])
            assert entry["Fcn_coulomb_negative"] == pytest.approx(theta_f[2 * n_dof + j_idx])

    def test_dynamics_tag_is_written_for_friction(self, tmp_path):
        """The URDF <dynamics> tag must reflect the identified Fv and |Fcp|/|Fcn| average."""
        import xml.etree.ElementTree as ET

        from src.urdf_exporter import export_adapted_urdf

        n_dof = 7
        pi_full, theta_f = self._make_synthetic_friction_pi(n_dof, "viscous_coulomb")
        out_urdf = tmp_path / "adapted_fr3.urdf"
        export_adapted_urdf(
            input_urdf_path=URDF_FR3,
            pi_full=pi_full,
            n_dof=n_dof,
            friction_model="viscous_coulomb",
            output_urdf_path=out_urdf,
            friction_sidecar_path=None,
        )
        root = ET.parse(out_urdf).getroot()
        revolute = [
            j for j in root.findall("joint")
            if j.get("type") in ("revolute", "continuous")
        ]
        assert len(revolute) == n_dof
        for j_idx, j_el in enumerate(revolute):
            dyn = j_el.find("dynamics")
            assert dyn is not None
            assert float(dyn.get("damping")) == pytest.approx(theta_f[j_idx])
            expected_friction = 0.5 * (
                abs(theta_f[n_dof + j_idx]) + abs(theta_f[2 * n_dof + j_idx])
            )
            assert float(dyn.get("friction")) == pytest.approx(expected_friction)

    def test_no_friction_means_no_sidecar(self, tmp_path):
        from src.kinematics import RobotKinematics
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_RRBOT)
        pi_full = RobotKinematics(robot).PI.flatten()
        out_urdf = tmp_path / "rrbot_no_friction.urdf"
        out_json = tmp_path / "would_not_be_written.json"
        meta = export_adapted_urdf(
            input_urdf_path=URDF_RRBOT,
            pi_full=pi_full,
            n_dof=robot.nDoF,
            friction_model="none",
            output_urdf_path=out_urdf,
            friction_sidecar_path=out_json,
        )
        assert meta["friction_sidecar_path"] is None
        assert not out_json.exists()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Defensive / validation paths
# ──────────────────────────────────────────────────────────────────────────────


class TestExportValidation:
    def test_non_positive_mass_raises(self, tmp_path):
        from src.kinematics import RobotKinematics
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_RRBOT)
        pi_full = RobotKinematics(robot).PI.flatten().copy()
        pi_full[0] = -1.0  # corrupt link 0 mass
        with pytest.raises(ValueError, match="non-positive"):
            export_adapted_urdf(
                input_urdf_path=URDF_RRBOT,
                pi_full=pi_full,
                n_dof=robot.nDoF,
                friction_model="none",
                output_urdf_path=tmp_path / "broken.urdf",
            )

    def test_parameter_length_mismatch_raises(self, tmp_path):
        from src.urdf_exporter import export_adapted_urdf

        with pytest.raises(ValueError, match="parameter length mismatch"):
            export_adapted_urdf(
                input_urdf_path=URDF_RRBOT,
                pi_full=np.ones(5),
                n_dof=2,
                friction_model="none",
                output_urdf_path=tmp_path / "broken.urdf",
            )

    def test_ndof_mismatch_raises(self, tmp_path):
        from src.kinematics import RobotKinematics
        from src.urdf_exporter import export_adapted_urdf
        from src.urdf_parser import parse_urdf

        robot = parse_urdf(URDF_RRBOT)
        pi_full = RobotKinematics(robot).PI.flatten()
        # Claim 3 DoF (matches the parameter length) while the URDF has 2 DoF.
        pi_full = np.concatenate([pi_full, pi_full[:10]])
        with pytest.raises(ValueError, match="nDoF"):
            export_adapted_urdf(
                input_urdf_path=URDF_RRBOT,
                pi_full=pi_full,
                n_dof=3,
                friction_model="none",
                output_urdf_path=tmp_path / "broken.urdf",
            )

    def test_config_loader_rejects_absolute_export_filename(self, tmp_path):
        from src.config_loader import load_config_dict

        urdf_copy = tmp_path / "rrbot.urdf"
        urdf_copy.write_bytes(Path(URDF_RRBOT).read_bytes())
        cfg = {
            "urdf_path": str(urdf_copy),
            "output_dir": str(tmp_path / "out"),
            "method": "newton_euler",
            "export": {
                "enabled": True,
                "urdf_filename": str(tmp_path / "abs.urdf"),
            },
        }
        with pytest.raises(ValueError, match="must be a plain filename"):
            load_config_dict(cfg, validate=True)

    def test_config_loader_rejects_parent_traversal(self, tmp_path):
        from src.config_loader import load_config_dict

        urdf_copy = tmp_path / "rrbot.urdf"
        urdf_copy.write_bytes(Path(URDF_RRBOT).read_bytes())
        cfg = {
            "urdf_path": str(urdf_copy),
            "output_dir": str(tmp_path / "out"),
            "method": "newton_euler",
            "export": {
                "enabled": True,
                "friction_sidecar_filename": "../escape.json",
            },
        }
        with pytest.raises(ValueError, match="must be a plain filename"):
            load_config_dict(cfg, validate=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Pipeline-level integration: Stage 12 fires when export.enabled = true
# ──────────────────────────────────────────────────────────────────────────────


def _elbow_pipeline_cfg(tmp_path: Path, *, export_block, feasibility="cholesky"):
    """Minimal Newton-Euler pipeline config for the 3-DoF elbow.

    The elbow is structurally underdetermined for unconstrained OLS (some
    link masses collapse to numerical zero, which the exporter rightly
    refuses).  We therefore default to ``feasibility_method="cholesky"`` so
    the integration tests exercise a physically feasible adapted URDF; the
    "exporter raises on non-positive mass" path is covered separately by
    :class:`TestExportValidation` against synthetic inputs.
    """
    cfg = {
        "urdf_path": URDF_ELBOW,
        "output_dir": str(tmp_path / "out"),
        "method": "newton_euler",
        "joint_limits": {
            "position": [[-3.14159, 3.14159], [-1.5708, 1.5708], [-1.5708, 1.5708]],
            "velocity": [[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]],
            "acceleration": [[-8.0, 8.0], [-8.0, 8.0], [-8.0, 8.0]],
        },
        "excitation": {
            "num_harmonics": 3,
            "base_frequency_hz": 0.5,
            "trajectory_duration_periods": 4,
            "optimize_condition_number": False,
            "optimizer_max_iter": 50,
        },
        "identification": {
            "solver": "ols",
            "feasibility_method": feasibility,
        },
    }
    if export_block is not None:
        cfg["export"] = export_block
    return cfg


class TestStage12Integration:
    def test_export_disabled_writes_no_extra_files(self, tmp_path):
        from src.pipeline import SystemIdentificationPipeline

        cfg = _elbow_pipeline_cfg(tmp_path, export_block=None)
        SystemIdentificationPipeline(cfg).run()
        out = tmp_path / "out"
        assert (out / "identification_results.npz").exists()
        # No adapted URDF produced when the export block is absent.
        assert not list(out.glob("adapted*.urdf"))
        allowed_json = {"results_summary.json", "regressor_model.json"}
        assert all(p.name in allowed_json for p in out.glob("*.json"))
        summary = json.loads((out / "results_summary.json").read_text())
        assert "export" not in summary

    def test_export_enabled_writes_adapted_urdf(self, tmp_path):
        from src.kinematics import RobotKinematics
        from src.pipeline import SystemIdentificationPipeline
        from src.urdf_parser import parse_urdf

        export_block = {
            "enabled": True,
            "urdf_filename": "adapted_elbow.urdf",
            "friction_sidecar": True,
            "friction_sidecar_filename": "adapted_friction.json",
        }
        cfg = _elbow_pipeline_cfg(tmp_path, export_block=export_block)
        SystemIdentificationPipeline(cfg).run()

        out = tmp_path / "out"
        adapted = out / "adapted_elbow.urdf"
        assert adapted.exists(), "Stage 12 must write the adapted URDF when enabled"
        # friction_model defaults to 'none' here, so no sidecar should be written.
        assert not (out / "adapted_friction.json").exists()

        results = np.load(str(out / "identification_results.npz"), allow_pickle=True)
        n_dof = int(results["nDoF"])
        pi_corrected = np.asarray(results["pi_corrected"], float).reshape(-1)

        adapted_robot = parse_urdf(str(adapted))
        adapted_kin = RobotKinematics(adapted_robot)
        np.testing.assert_allclose(
            adapted_kin.PI.flatten(),
            pi_corrected[: 10 * n_dof],
            atol=1e-7,
            rtol=1e-7,
        )

        summary = json.loads((out / "results_summary.json").read_text())
        assert "export" in summary
        assert summary["export"]["adapted_urdf_path"].endswith("adapted_elbow.urdf")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Unified runner threads the export block through to the pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestRunnerThreading:
    """The ``export`` block in a unified config must reach Stage 12 unchanged."""

    def test_unified_runner_writes_adapted_urdf(self, tmp_path):
        from src.runner import UnifiedRunner

        unified_cfg = {
            "urdf_path": URDF_ELBOW,
            "output_dir": str(tmp_path / "out"),
            "method": "newton_euler",
            "stages": {
                "excitation": True,
                "identification": True,
                "validation_pybullet": False,
                "report": False,
                "benchmark": False,
                "plot": False,
            },
            "resume": {"from_checkpoint": None},
            "joint_limits": {
                "position": [[-3.14159, 3.14159], [-1.5708, 1.5708], [-1.5708, 1.5708]],
                "velocity": [[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]],
                "acceleration": [[-8.0, 8.0], [-8.0, 8.0], [-8.0, 8.0]],
            },
            "excitation": {
                "num_harmonics": 3,
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 4,
                "optimize_condition_number": False,
                "optimizer_max_iter": 50,
            },
            "identification": {
                "solver": "ols",
                "feasibility_method": "cholesky",
                "data_file": None,
            },
            "export": {
                "enabled": True,
                "urdf_filename": "via_runner.urdf",
                "friction_sidecar": True,
                "friction_sidecar_filename": "via_runner_friction.json",
            },
        }
        cfg_path = tmp_path / "unified.json"
        cfg_path.write_text(json.dumps(unified_cfg, indent=2), encoding="utf-8")

        rc = UnifiedRunner(str(cfg_path)).run()
        assert rc == 0

        adapted = tmp_path / "out" / "pipeline" / "via_runner.urdf"
        assert adapted.exists(), (
            "UnifiedRunner must thread the 'export' block through to the "
            "pipeline so Stage 12 writes the adapted URDF"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. Standalone CLI tool still works (delegates to src.urdf_exporter)
# ──────────────────────────────────────────────────────────────────────────────


class TestStandaloneCLIDelegates:
    def test_tool_exports_from_npz(self, tmp_path, monkeypatch):
        from src.kinematics import RobotKinematics
        from src.urdf_parser import parse_urdf

        # Spec the tool by importing it as a module. The CLI lives next to
        # the exporter under src/ so it can stay a single-file script while
        # still resolving ``from src.urdf_exporter import ...`` at run time.
        tool_path = ROOT / "src" / "export_adapted_urdf.py"
        spec = importlib.util.spec_from_file_location("export_adapted_urdf", tool_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        robot = parse_urdf(URDF_RRBOT)
        pi_full = RobotKinematics(robot).PI.flatten()

        npz_path = tmp_path / "identification_results.npz"
        np.savez(
            str(npz_path),
            pi_corrected=pi_full,
            nDoF=np.int64(robot.nDoF),
            friction_model=np.asarray("none"),
        )
        out_urdf = tmp_path / "tool_out.urdf"
        module.export_adapted(
            in_urdf=Path(URDF_RRBOT),
            in_npz=npz_path,
            out_urdf=out_urdf,
            out_friction_json=None,
        )
        assert out_urdf.exists()
        kin2 = RobotKinematics(parse_urdf(str(out_urdf)))
        np.testing.assert_allclose(kin2.PI.flatten(), pi_full, atol=1e-9, rtol=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Slow PyBullet round-trip — adapted URDF must be dynamically equivalent
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_pybullet_round_trip_consistency(tmp_path):
    """Re-export a feasible URDF and verify PyBullet inverse dynamics match.

    The original URDF is parsed into its Atkeson 10-vector via
    :class:`RobotKinematics`, then run through :func:`export_adapted_urdf`.
    PyBullet inverse-dynamics torques on the *adapted* URDF must match the
    inverse-dynamics torques on the *original* URDF to floating-point
    precision, since the round-trip is mathematically lossless.

    This decouples the exporter check from the identification quality and
    is the strongest available verification that the parallel-axis inverse
    in :func:`_unpack_link_inertials` is correct.

    Skipped unless ``--run-slow`` is passed and ``pybullet`` is importable.
    """
    if importlib.util.find_spec("pybullet") is None:
        pytest.skip("pybullet not installed")

    import pybullet as pb

    from src.kinematics import RobotKinematics
    from src.urdf_exporter import export_adapted_urdf
    from src.urdf_parser import parse_urdf

    src_urdf = URDF_FR3
    robot = parse_urdf(src_urdf)
    n_dof = robot.nDoF
    pi_full = RobotKinematics(robot).PI.flatten()

    adapted = tmp_path / "adapted_fr3.urdf"
    export_adapted_urdf(
        input_urdf_path=src_urdf,
        pi_full=pi_full,
        n_dof=n_dof,
        friction_model="none",
        output_urdf_path=adapted,
    )

    rng = np.random.default_rng(seed=0)
    q = rng.uniform(-0.4, 0.4, size=n_dof)
    dq = rng.uniform(-0.5, 0.5, size=n_dof)
    ddq = rng.uniform(-1.0, 1.0, size=n_dof)

    def _pb_inverse_dynamics(urdf_path: str) -> np.ndarray:
        client = pb.connect(pb.DIRECT)
        try:
            body = pb.loadURDF(urdf_path, useFixedBase=True, physicsClientId=client)
            joint_indices = [
                i for i in range(pb.getNumJoints(body, physicsClientId=client))
                if pb.getJointInfo(body, i, physicsClientId=client)[2]
                == pb.JOINT_REVOLUTE
            ]
            assert len(joint_indices) == n_dof
            for idx, ji in enumerate(joint_indices):
                pb.resetJointState(body, ji, q[idx], dq[idx], physicsClientId=client)
            tau = pb.calculateInverseDynamics(
                body, q.tolist(), dq.tolist(), ddq.tolist(),
                physicsClientId=client,
            )
            return np.asarray(tau)
        finally:
            pb.disconnect(client)

    tau_original = _pb_inverse_dynamics(src_urdf)
    tau_adapted = _pb_inverse_dynamics(str(adapted))

    np.testing.assert_allclose(tau_adapted, tau_original, atol=1e-9, rtol=1e-9)
