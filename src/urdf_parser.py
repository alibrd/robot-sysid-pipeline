"""URDF/XACRO XML parser: extracts joint chain, inertial parameters, and joint limits."""
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .math_utils import rpy_to_rotation_np, make_transform_np


@dataclass
class JointData:
    name: str
    joint_type: str  # "fixed" | "revolute" | "continuous" | "prismatic"
    parent: str
    child: str
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    limit_lower: Optional[float] = None
    limit_upper: Optional[float] = None
    limit_velocity: Optional[float] = None
    limit_effort: Optional[float] = None


@dataclass
class InertialData:
    mass: float = 0.0
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0

    @property
    def tensor(self) -> np.ndarray:
        return np.array([
            [self.ixx, self.ixy, self.ixz],
            [self.ixy, self.iyy, self.iyz],
            [self.ixz, self.iyz, self.izz]
        ])


@dataclass
class LinkData:
    name: str
    inertial: InertialData = field(default_factory=InertialData)


@dataclass
class RobotDescription:
    """Structured output of the URDF parser."""
    name: str
    joints: List[JointData]
    links: dict  # name -> LinkData
    nDoF: int
    revolute_joint_names: List[str]
    Tw_0: np.ndarray  # world-to-first-revolute fixed transform
    chain_joints: List[JointData]  # ordered serial chain (fixed + revolute)


def parse_urdf(urdf_path: str) -> RobotDescription:
    """Parse a URDF (or .xacro) file and return structured robot description."""
    resolved = _resolve_xacro(urdf_path)
    tree = ET.parse(resolved)
    root = tree.getroot()
    robot_name = root.get("name", "unnamed_robot")

    # -- Parse all joints ----------------------------------------------------
    joints: List[JointData] = []
    for jel in root.findall("joint"):
        jd = JointData(
            name=jel.get("name"),
            joint_type=jel.get("type"),
            parent=jel.find("parent").get("link"),
            child=jel.find("child").get("link"),
        )
        origin = jel.find("origin")
        if origin is not None:
            xyz_str = origin.get("xyz")
            if xyz_str:
                jd.origin_xyz = np.array([float(x) for x in xyz_str.split()])
            rpy_str = origin.get("rpy")
            if rpy_str:
                jd.origin_rpy = np.array([float(x) for x in rpy_str.split()])

        axis_el = jel.find("axis")
        if axis_el is not None:
            ax_str = axis_el.get("xyz")
            if ax_str:
                jd.axis = np.array([float(x) for x in ax_str.split()])

        limit_el = jel.find("limit")
        if limit_el is not None:
            lo = limit_el.get("lower")
            hi = limit_el.get("upper")
            vel = limit_el.get("velocity")
            eff = limit_el.get("effort")
            jd.limit_lower = float(lo) if lo is not None else None
            jd.limit_upper = float(hi) if hi is not None else None
            jd.limit_velocity = float(vel) if vel is not None else None
            jd.limit_effort = float(eff) if eff is not None else None

        joints.append(jd)

    # -- Parse all links (inertial data) -------------------------------------
    links = {}
    for lel in root.findall("link"):
        lname = lel.get("name")
        ld = LinkData(name=lname)
        iner = lel.find("inertial")
        if iner is not None:
            mass_el = iner.find("mass")
            if mass_el is not None:
                ld.inertial.mass = float(mass_el.get("value", "0"))
            origin_el = iner.find("origin")
            if origin_el is not None:
                xyz_str = origin_el.get("xyz")
                if xyz_str:
                    ld.inertial.origin_xyz = np.array([float(x) for x in xyz_str.split()])
                rpy_str = origin_el.get("rpy")
                if rpy_str:
                    ld.inertial.origin_rpy = np.array([float(x) for x in rpy_str.split()])
            inertia_el = iner.find("inertia")
            if inertia_el is not None:
                for attr in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    val = inertia_el.get(attr)
                    if val is not None:
                        setattr(ld.inertial, attr, float(val))
        links[lname] = ld

    # -- Build topology-based serial chain -----------------------------------
    # Index joints by parent link for O(1) lookup
    children_of = {}  # parent_link_name -> list of JointData
    all_child_links = set()
    for j in joints:
        children_of.setdefault(j.parent, []).append(j)
        all_child_links.add(j.child)

    # Find root link (a link that is not a child of any joint)
    all_link_names = set(links.keys())
    root_links = all_link_names - all_child_links
    if not root_links:
        raise ValueError("Could not find a root link in the URDF (cycle or empty).")
    root_link = sorted(root_links)[0]  # deterministic pick

    # Walk from root link to build the serial chain
    chain_joints: List[JointData] = []
    current_link = root_link
    while current_link in children_of:
        child_joints = children_of[current_link]
        if len(child_joints) != 1:
            # For branching robots, pick the branch that has the most revolute joints
            # (simple heuristic for serial chain extraction)
            best = max(child_joints, key=lambda j: _count_revolute_descendants(j.child, children_of, joints))
            child_joints = [best]
        jd = child_joints[0]
        chain_joints.append(jd)
        current_link = jd.child

    # -- Build Tw_0: fixed transforms BEFORE the first revolute joint --------
    Tw_0 = np.eye(4)
    revolute_names = []
    for j in chain_joints:
        if j.joint_type in ("revolute", "continuous"):
            revolute_names.append(j.name)
            break  # stop accumulating fixed transforms at the first revolute
        elif j.joint_type == "fixed":
            R0 = rpy_to_rotation_np(*j.origin_rpy)
            Tw_0 = Tw_0 @ make_transform_np(R0, j.origin_xyz)

    # Collect all revolute joint names from the chain
    revolute_names = [j.name for j in chain_joints if j.joint_type in ("revolute", "continuous")]
    nDoF = len(revolute_names)

    return RobotDescription(
        name=robot_name,
        joints=joints,
        links=links,
        nDoF=nDoF,
        revolute_joint_names=revolute_names,
        Tw_0=Tw_0,
        chain_joints=chain_joints,
    )


def _count_revolute_descendants(link_name, children_of, joints):
    """Count revolute joints reachable from *link_name* (for chain selection)."""
    count = 0
    current = link_name
    while current in children_of:
        child_joints = children_of[current]
        for j in child_joints:
            if j.joint_type in ("revolute", "continuous"):
                count += 1
        if len(child_joints) == 1:
            current = child_joints[0].child
        else:
            break
    return count


def extract_joint_limits(robot: RobotDescription, cfg_limits: dict, logger):
    """Extract joint limits from URDF or config, raising if missing.

    Returns (q_lim, dq_lim, ddq_lim) each as (nDoF, 2) arrays [lower, upper].
    """
    n = robot.nDoF

    # ── Position limits ──
    q_lim = np.zeros((n, 2))
    cfg_pos = cfg_limits.get("position")
    for i, jname in enumerate(robot.revolute_joint_names):
        jd = next(j for j in robot.joints if j.name == jname)
        if cfg_pos is not None and i < len(cfg_pos):
            q_lim[i] = cfg_pos[i]
        elif jd.limit_lower is not None and jd.limit_upper is not None:
            q_lim[i] = [jd.limit_lower, jd.limit_upper]
        else:
            raise ValueError(
                f"Position limits missing for joint '{jname}'. "
                "Specify them in the JSON config under joint_limits.position."
            )
    logger.debug("Position limits:\n%s", q_lim)

    # ── Velocity limits ──
    dq_lim = np.zeros((n, 2))
    cfg_vel = cfg_limits.get("velocity")
    for i, jname in enumerate(robot.revolute_joint_names):
        jd = next(j for j in robot.joints if j.name == jname)
        if cfg_vel is not None and i < len(cfg_vel):
            dq_lim[i] = cfg_vel[i]
        elif jd.limit_velocity is not None:
            dq_lim[i] = [-jd.limit_velocity, jd.limit_velocity]
        else:
            raise ValueError(
                f"Velocity limits missing for joint '{jname}'. "
                "Specify them in the JSON config under joint_limits.velocity."
            )
    logger.debug("Velocity limits:\n%s", dq_lim)

    # ── Acceleration limits ──
    ddq_lim = np.zeros((n, 2))
    cfg_acc = cfg_limits.get("acceleration")
    for i, jname in enumerate(robot.revolute_joint_names):
        if cfg_acc is not None and i < len(cfg_acc):
            ddq_lim[i] = cfg_acc[i]
        else:
            raise ValueError(
                f"Acceleration limits missing for joint '{jname}'. "
                "Specify them in the JSON config under joint_limits.acceleration."
            )
    logger.debug("Acceleration limits:\n%s", ddq_lim)

    return q_lim, dq_lim, ddq_lim


def _resolve_xacro(urdf_path: str) -> str:
    """If *urdf_path* ends with ``.xacro``, run ``xacro`` to produce URDF XML.

    Returns the path to a plain URDF (either the original or a temp file).
    """
    p = Path(urdf_path)
    if p.suffix.lower() != ".xacro":
        return str(p)

    try:
        result = subprocess.run(
            ["xacro", str(p)],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Cannot process .xacro file '{p}' -- the 'xacro' command is not "
            "available. Install it with: pip install xacro"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"xacro failed on '{p}':\n{exc.stderr}") from exc

    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(result.stdout)
    tmp.close()
    return tmp.name
