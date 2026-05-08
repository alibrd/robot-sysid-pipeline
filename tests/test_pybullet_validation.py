"""Tests for the standalone PyBullet validation workflow."""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "tests" / "assets"
URDF_RRBOT = str(ASSET_DIR / "RRBot_single.urdf")
URDF_PENDULUM = str(ASSET_DIR / "DrakePendulum_1DoF.urdf")
URDF_FINGEREDU = str(ASSET_DIR / "FingerEdu_3DoF.xacro")


def _write_excitation(path: Path, params, freqs, q0, basis="cosine", optimize_phase=False):
    np.savez(
        str(path),
        params=np.asarray(params, dtype=float),
        freqs=np.asarray(freqs, dtype=float),
        q0=np.asarray(q0, dtype=float),
        basis=np.array(basis),
        optimize_phase=np.array(optimize_phase),
    )


def _write_validation_config(path: Path,
                             urdf_path: str,
                             excitation_file: str,
                             output_dir: str,
                             base_frequency_hz: float,
                             trajectory_duration_periods: float,
                             sample_rate_hz: float = 0.0,
                             tolerance_abs: float = 1e-6,
                             tolerance_normalized_rms: float = 1e-6,
                             gravity=None,
                             use_deprecated_tolerance_key: bool = False):
    comparison = {
        "tolerance_abs": tolerance_abs,
    }
    tolerance_key = (
        "tolerance_rel"
        if use_deprecated_tolerance_key
        else "tolerance_normalized_rms"
    )
    comparison[tolerance_key] = tolerance_normalized_rms
    cfg = {
        "urdf_path": urdf_path,
        "excitation_file": excitation_file,
        "output_dir": output_dir,
        "base_frequency_hz": base_frequency_hz,
        "trajectory_duration_periods": trajectory_duration_periods,
        "sample_rate_hz": sample_rate_hz,
        "gravity": [0.0, 0.0, -9.80665] if gravity is None else gravity,
        "comparison": comparison,
    }
    path.write_text(json.dumps(cfg), encoding="utf-8")


def test_replay_excitation_trajectory_matches_fourier_reconstruction(tmp_path):
    from src.pybullet_validation import replay_excitation_trajectory
    from src.trajectory import fourier_trajectory

    excitation_path = tmp_path / "excitation.npz"
    _write_excitation(
        excitation_path,
        params=[0.1, -0.2],
        freqs=[0.5],
        q0=[0.05, -0.1],
        basis="cosine",
    )

    t, q, dq, ddq, sample_rate, artifact = replay_excitation_trajectory(
        str(excitation_path), 0.5, 1.0, 200.0
    )

    q_ref, dq_ref, ddq_ref = fourier_trajectory(
        artifact["params"], artifact["freqs"], t, artifact["q0"],
        artifact["basis"], artifact["optimize_phase"]
    )
    np.testing.assert_allclose(q, q_ref.T)
    np.testing.assert_allclose(dq, dq_ref.T)
    np.testing.assert_allclose(ddq, ddq_ref.T)
    assert sample_rate == 200.0


@pytest.mark.parametrize(
    ("basis", "optimize_phase", "params"),
    [
        ("sine", False, [0.1, -0.2]),
        ("both", False, [0.1, -0.2, 0.03, -0.04]),
        ("both", True, [0.1, -0.2, 0.25, -0.35]),
    ],
)
def test_replay_excitation_trajectory_supports_sine_and_both_bases(
        tmp_path, basis, optimize_phase, params):
    from src.pybullet_validation import replay_excitation_trajectory
    from src.trajectory import fourier_trajectory

    excitation_path = tmp_path / f"excitation_{basis}_{int(optimize_phase)}.npz"
    _write_excitation(
        excitation_path,
        params=params,
        freqs=[0.5],
        q0=[0.05, -0.1],
        basis=basis,
        optimize_phase=optimize_phase,
    )

    t, q, dq, ddq, _, artifact = replay_excitation_trajectory(
        str(excitation_path), 0.5, 1.0, 200.0
    )

    q_ref, dq_ref, ddq_ref = fourier_trajectory(
        artifact["params"], artifact["freqs"], t, artifact["q0"],
        artifact["basis"], artifact["optimize_phase"]
    )
    np.testing.assert_allclose(q, q_ref.T)
    np.testing.assert_allclose(dq, dq_ref.T)
    np.testing.assert_allclose(ddq, ddq_ref.T)


def test_compute_comparison_metrics_handles_near_zero_reference():
    from src.pybullet_validation import compute_comparison_metrics

    tau_reference = np.array([[0.0, 1.0], [0.0, -1.0]])
    tau_candidate = np.array([[1e-10, 1.001], [2e-10, -0.999]])
    metrics = compute_comparison_metrics(
        tau_reference,
        tau_candidate,
        tolerance_abs=0.01,
        tolerance_normalized_rms=0.01,
    )

    assert metrics["passed"]
    assert np.isfinite(metrics["tau_rel_error"]).all()
    assert metrics["global_max_abs_error"] <= 0.0010001


def test_compute_torques_rejects_joint_order_mismatch(monkeypatch):
    from src.pybullet_validation import compute_torques
    monkeypatch.setattr(
        "src.pybullet_validation._prepare_pybullet_urdf",
        lambda urdf_path: (urdf_path, None),
    )

    class FakePyBullet:
        DIRECT = 0
        JOINT_FIXED = 4
        JOINT_REVOLUTE = 0
        URDF_USE_INERTIA_FROM_FILE = 1

        @staticmethod
        def connect(mode):
            return 1

        @staticmethod
        def setGravity(*args, **kwargs):
            return None

        @staticmethod
        def loadURDF(*args, **kwargs):
            return 7

        @staticmethod
        def getNumJoints(*args, **kwargs):
            return 2

        @staticmethod
        def getJointInfo(body_id, joint_index, physicsClientId=None):
            names = [b"joint_a", b"joint_b"]
            return (joint_index, names[joint_index], FakePyBullet.JOINT_REVOLUTE)

        @staticmethod
        def changeDynamics(*args, **kwargs):
            return None

        @staticmethod
        def calculateInverseDynamics(*args, **kwargs):
            return [0.0, 0.0]

        @staticmethod
        def disconnect(*args, **kwargs):
            return None

    monkeypatch.setitem(sys.modules, "pybullet", FakePyBullet)
    with pytest.raises(ValueError, match="joint order"):
        compute_torques(
            "dummy.urdf",
            ["joint_b", "joint_a"],
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            [0.0, 0.0, -9.80665],
        )


@pytest.mark.skipif(importlib.util.find_spec("pybullet") is None,
                    reason="pybullet is not installed")
def test_pybullet_validation_runner_pendulum_1dof(tmp_path):
    from src.pybullet_validation import PyBulletValidationRunner
    from src.urdf_parser import parse_urdf

    excitation_path = tmp_path / "excitation.npz"
    config_path = tmp_path / "cfg.json"
    output_dir = tmp_path / "out"

    _write_excitation(
        excitation_path,
        params=[0.05],
        freqs=[0.5],
        q0=[0.0],
        basis="cosine",
    )
    _write_validation_config(
        config_path,
        URDF_PENDULUM,
        str(excitation_path),
        str(output_dir),
        0.5,
        1.0,
        sample_rate_hz=100.0,
        tolerance_abs=1e-4,
        tolerance_normalized_rms=1e-4,
    )

    summary = PyBulletValidationRunner(str(config_path)).run()
    robot_name = parse_urdf(URDF_PENDULUM).name

    assert summary["passed"]
    assert (output_dir / robot_name / "pybullet_validation_summary.json").exists()
    assert (output_dir / robot_name / "pybullet_validation_data.npz").exists()
    assert (output_dir / robot_name / "pybullet_validation.log").exists()


@pytest.mark.skipif(importlib.util.find_spec("pybullet") is None,
                    reason="pybullet is not installed")
def test_pybullet_validation_runner_rrbot(tmp_path):
    from src.pybullet_validation import PyBulletValidationRunner

    excitation_path = tmp_path / "excitation_rrbot.npz"
    config_path = tmp_path / "cfg_rrbot.json"
    output_dir = tmp_path / "out_rrbot"

    _write_excitation(
        excitation_path,
        params=[0.03, -0.02],
        freqs=[0.5],
        q0=[0.0, 0.1],
        basis="cosine",
    )
    _write_validation_config(
        config_path,
        URDF_RRBOT,
        str(excitation_path),
        str(output_dir),
        0.5,
        1.0,
        sample_rate_hz=100.0,
        tolerance_abs=1e-4,
        tolerance_normalized_rms=1e-4,
    )

    summary = PyBulletValidationRunner(str(config_path)).run()

    assert summary["passed"]
    assert (output_dir / "single_rrbot" / "pybullet_validation_summary.json").exists()
    assert (output_dir / "single_rrbot" / "pybullet_validation_data.npz").exists()
    assert (output_dir / "single_rrbot" / "pybullet_validation.log").exists()


def test_pybullet_validation_runner_rejects_gravity_override(tmp_path):
    from src.pybullet_validation import PyBulletValidationRunner

    excitation_path = tmp_path / "excitation.npz"
    config_path = tmp_path / "cfg.json"
    output_dir = tmp_path / "out"

    _write_excitation(
        excitation_path,
        params=[0.05],
        freqs=[0.5],
        q0=[0.0],
        basis="cosine",
    )
    _write_validation_config(
        config_path,
        URDF_PENDULUM,
        str(excitation_path),
        str(output_dir),
        0.5,
        1.0,
        sample_rate_hz=100.0,
        gravity=[0.0, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="hardcoded math_utils\\.GRAVITY"):
        PyBulletValidationRunner(str(config_path)).run()


def test_reorder_columns_supports_explicit_joint_reordering():
    from src.pybullet_validation import _reorder_columns

    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    reordered = _reorder_columns(
        values,
        source_names=["joint_a", "joint_b"],
        target_names=["joint_b", "joint_a"],
    )

    np.testing.assert_array_equal(reordered, values[:, ::-1])


def test_prepare_pybullet_urdf_patches_continuous_joint_limits(tmp_path):
    from src.pybullet_validation import _prepare_pybullet_urdf

    urdf_path = tmp_path / "continuous_joint.urdf"
    urdf_path.write_text(
        """
<robot name="continuous_test">
  <link name="base"/>
  <link name="tip"/>
  <joint name="joint_1" type="continuous">
    <parent link="base"/>
    <child link="tip"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
""".strip(),
        encoding="utf-8",
    )

    patched_path, temp_path = _prepare_pybullet_urdf(str(urdf_path))
    try:
        assert patched_path != str(urdf_path)
        assert temp_path is not None

        root = ET.parse(patched_path).getroot()
        limit_el = root.find("./joint[@name='joint_1']/limit")
        assert limit_el is not None
        assert limit_el.get("lower") == "-1e30"
        assert limit_el.get("upper") == "1e30"
        assert limit_el.get("effort") == "1e30"
        assert limit_el.get("velocity") == "1e30"
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def test_prepare_pybullet_urdf_resolves_fingeredu_xacro_and_rewrites_meshes():
    from src.pybullet_validation import _prepare_pybullet_urdf

    patched_path, temp_path = _prepare_pybullet_urdf(URDF_FINGEREDU)
    try:
        assert temp_path is not None

        root = ET.parse(patched_path).getroot()
        assert root.get("name") == "finger"

        mesh_files = [
            mesh.get("filename")
            for mesh in root.findall(".//mesh")
            if mesh.get("filename") is not None
        ]
        assert mesh_files
        assert all(not filename.startswith("package://") for filename in mesh_files)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def test_validation_config_deprecated_tolerance_rel_warns_and_migrates(tmp_path):
    from src.pybullet_validation import load_pybullet_validation_config

    excitation_path = tmp_path / "excitation.npz"
    config_path = tmp_path / "cfg.json"

    _write_excitation(
        excitation_path,
        params=[0.05],
        freqs=[0.5],
        q0=[0.0],
        basis="cosine",
    )
    _write_validation_config(
        config_path,
        URDF_PENDULUM,
        str(excitation_path),
        str(tmp_path / "out"),
        0.5,
        1.0,
        tolerance_abs=1e-4,
        tolerance_normalized_rms=2e-4,
        use_deprecated_tolerance_key=True,
    )

    with pytest.warns(DeprecationWarning, match="tolerance_rel"):
        cfg = load_pybullet_validation_config(str(config_path))

    assert cfg["comparison"]["tolerance_normalized_rms"] == pytest.approx(2e-4)
    assert "tolerance_rel" not in cfg["comparison"]


@pytest.mark.skipif(importlib.util.find_spec("pybullet") is None,
                    reason="pybullet is not installed")
def test_pybullet_validation_runner_reports_fail_for_wrong_inertia(tmp_path, monkeypatch):
    from src.kinematics import RobotKinematics as BaseRobotKinematics
    from src.pybullet_validation import PyBulletValidationRunner

    class PerturbedRobotKinematics(BaseRobotKinematics):
        def __init__(self, robot, logger=None):
            super().__init__(robot, logger)
            self.PI = self.PI.copy()
            self.PI[:10, 0] *= 1.5

    monkeypatch.setattr(
        "src.pybullet_validation.RobotKinematics",
        PerturbedRobotKinematics,
    )

    excitation_path = tmp_path / "excitation_fail.npz"
    config_path = tmp_path / "cfg_fail.json"
    output_dir = tmp_path / "out_fail"

    _write_excitation(
        excitation_path,
        params=[0.05],
        freqs=[0.5],
        q0=[0.0],
        basis="cosine",
    )
    _write_validation_config(
        config_path,
        URDF_PENDULUM,
        str(excitation_path),
        str(output_dir),
        0.5,
        1.0,
        sample_rate_hz=100.0,
        tolerance_abs=1e-4,
        tolerance_normalized_rms=1e-4,
    )

    summary = PyBulletValidationRunner(str(config_path)).run()

    assert not summary["passed"]
    assert summary["global_max_abs_error"] > summary["tolerance_abs"]


def test_replay_excitation_rejects_frequency_mismatch(tmp_path):
    from src.pybullet_validation import replay_excitation_trajectory

    excitation_path = tmp_path / "excitation_mismatch.npz"
    _write_excitation(
        excitation_path,
        params=[0.1, -0.2],
        freqs=[0.5],
        q0=[0.05, -0.1],
        basis="cosine",
    )

    with pytest.raises(ValueError, match="frequencies do not match"):
        replay_excitation_trajectory(
            str(excitation_path), 0.3, 1.0, 200.0
        )


# ──────────────────────────────────────────────────────────────────────────────
# Validation config loader path resolution
# ──────────────────────────────────────────────────────────────────────────────

def test_validation_config_loader_resolves_relative_paths(tmp_path):
    """PyBullet validation config loader must resolve URDF, excitation, output_dir."""
    import shutil
    from src.pybullet_validation import load_pybullet_validation_config

    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True, exist_ok=True)
    urdf_copy = tmp_path / "assets" / "DrakePendulum_1DoF.urdf"
    urdf_copy.parent.mkdir(parents=True)
    shutil.copy2(URDF_PENDULUM, urdf_copy)

    excitation_file = tmp_path / "data" / "excitation.npz"
    excitation_file.parent.mkdir(parents=True)
    np.savez(
        str(excitation_file),
        params=np.asarray([0.05], dtype=float),
        freqs=np.asarray([0.5], dtype=float),
        q0=np.asarray([0.0], dtype=float),
        basis=np.array("cosine"),
        optimize_phase=np.array(False),
    )

    config_path = config_dir / "validation.json"
    config_path.write_text(
        json.dumps(
            {
                "urdf_path": "../assets/DrakePendulum_1DoF.urdf",
                "excitation_file": "../data/excitation.npz",
                "output_dir": "../validation_output",
                "base_frequency_hz": 0.5,
                "trajectory_duration_periods": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cfg = load_pybullet_validation_config(str(config_path))

    assert cfg["urdf_path"] == str(
        (config_dir / "../assets/DrakePendulum_1DoF.urdf").resolve()
    )
    assert cfg["excitation_file"] == str(excitation_file.resolve())
    assert cfg["output_dir"] == str((config_dir / "../validation_output").resolve())
