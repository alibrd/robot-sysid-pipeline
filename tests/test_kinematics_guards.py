"""Guard-rail tests: unsupported joint types and skew axes must raise.

Silently mishandling these produced wrong dynamics before; the kinematics
builder now rejects them with actionable errors.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="guard_test">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0.1"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="j1" type="{joint_type}">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="{axis}"/>
    <limit lower="-1" upper="1" velocity="2" effort="10"/>
  </joint>
</robot>
"""


def _build_kinematics(tmp_path, *, joint_type="revolute", axis="0 0 1"):
    from src.kinematics import RobotKinematics
    from src.urdf_parser import parse_urdf

    urdf = tmp_path / "guard_test.urdf"
    urdf.write_text(
        _URDF_TEMPLATE.format(joint_type=joint_type, axis=axis),
        encoding="utf-8",
    )
    return RobotKinematics(parse_urdf(str(urdf)))


def test_axis_aligned_axes_accepted(tmp_path):
    for axis in ("0 0 1", "0 0 -1", "0 1 0", "1 0 0", "0 0 2"):
        kin = _build_kinematics(tmp_path, axis=axis)
        assert kin.nDoF == 1


def test_skew_axis_rejected(tmp_path):
    with pytest.raises(ValueError, match="not aligned with a coordinate axis"):
        _build_kinematics(tmp_path, axis="0.7071 0 0.7071")


def test_zero_axis_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid axis"):
        _build_kinematics(tmp_path, axis="0 0 0")


def test_prismatic_joint_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match="prismatic"):
        _build_kinematics(tmp_path, joint_type="prismatic")
