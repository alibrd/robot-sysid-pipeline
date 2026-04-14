"""Standalone PyBullet-based torque consistency validation."""
import importlib
import json
import tempfile
import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

from .config_utils import deep_merge, resolve_path_value
from .dynamics_newton_euler import newton_euler_regressor
from .kinematics import RobotKinematics
from .math_utils import GRAVITY
from .pipeline_logger import setup_logger
from .trajectory import build_frequencies, fourier_trajectory
from .urdf_parser import parse_urdf


def load_default_pybullet_validation_config() -> dict:
    """Load the repository default PyBullet validation config."""
    default_path = (Path(__file__).resolve().parent.parent
                    / "config" / "default_pybullet_validation_config.json")
    with open(default_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_pybullet_validation_config(config_path: str) -> dict:
    """Load the validation config, merge defaults, resolve paths, and validate."""
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8-sig") as f:
        user_cfg = json.load(f)

    return load_pybullet_validation_config_dict(
        user_cfg,
        config_path=str(config_file),
        resolve_relative_to=config_file.parent,
        validate=True,
    )


def load_pybullet_validation_config_dict(user_cfg: dict,
                                         config_path: str = "<inline>",
                                         resolve_relative_to: str | Path | None = None,
                                         validate: bool = True) -> dict:
    """Merge a validation config dict with defaults, resolve paths, and validate."""
    defaults = load_default_pybullet_validation_config()
    normalized_user_cfg = normalize_pybullet_validation_config_aliases(
        user_cfg,
        config_path=config_path,
    )
    cfg = deep_merge(defaults, normalized_user_cfg)
    cfg = _resolve_paths(cfg, resolve_relative_to)
    if validate:
        _validate_config(cfg, config_path)
    return cfg


def normalize_pybullet_validation_config_aliases(user_cfg: dict,
                                                 config_path: str = "<inline>") -> dict:
    """Normalize deprecated validation config aliases into the canonical schema."""
    normalized = deepcopy(user_cfg)
    comparison = normalized.get("comparison")
    if isinstance(comparison, dict) and "tolerance_rel" in comparison:
        warnings.warn(
            "comparison.tolerance_rel is deprecated; use "
            "comparison.tolerance_normalized_rms instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        comparison.setdefault(
            "tolerance_normalized_rms",
            comparison["tolerance_rel"],
        )
        comparison.pop("tolerance_rel", None)
    return normalized


def replay_excitation_trajectory(excitation_file: str,
                                 base_frequency_hz: float,
                                 trajectory_duration_periods: float,
                                 sample_rate_hz: float = 0.0):
    """Replay an excitation artifact into q, dq, ddq arrays."""
    artifact = _load_excitation_artifact(excitation_file)
    freqs = artifact["freqs"]
    expected_freqs = build_frequencies(base_frequency_hz, freqs.size)
    if not np.allclose(freqs, expected_freqs, atol=1e-12):
        raise ValueError(
            "Excitation frequencies do not match the provided base_frequency_hz "
            f"({base_frequency_hz}). Expected {expected_freqs.tolist()}, got "
            f"{freqs.tolist()}."
        )

    sample_rate = float(sample_rate_hz)
    if sample_rate <= 0.0:
        sample_rate = max(100.0, 20.0 * float(np.max(freqs)))

    duration = float(trajectory_duration_periods) / float(base_frequency_hz)
    t = np.arange(0.0, duration, 1.0 / sample_rate)
    if t.size == 0:
        raise ValueError("The validation time vector is empty. Increase sample_rate_hz.")

    q_t, dq_t, ddq_t = fourier_trajectory(
        artifact["params"],
        freqs,
        t,
        artifact["q0"],
        artifact["basis"],
        artifact["optimize_phase"],
    )
    q = q_t.T
    dq = dq_t.T
    ddq = ddq_t.T
    return t, q, dq, ddq, sample_rate, artifact


def compute_torques(urdf_path: str,
                    joint_names,
                    q: np.ndarray,
                    dq: np.ndarray,
                    ddq: np.ndarray,
                    gravity,
                    fixed_base: bool = True) -> np.ndarray:
    """Compute inverse-dynamics torques from PyBullet for a given trajectory."""
    p = _import_pybullet()
    q = np.asarray(q, dtype=float)
    dq = np.asarray(dq, dtype=float)
    ddq = np.asarray(ddq, dtype=float)
    gravity = np.asarray(gravity, dtype=float)
    joint_names = [str(name) for name in joint_names]

    if q.ndim != 2 or dq.shape != q.shape or ddq.shape != q.shape:
        raise ValueError("q, dq, and ddq must all have shape (N, nDoF).")
    if q.shape[1] != len(joint_names):
        raise ValueError("The joint_names length must match the trajectory width.")
    if gravity.shape != (3,):
        raise ValueError("gravity must contain exactly 3 elements.")

    client = p.connect(p.DIRECT)
    loadable_urdf_path, temp_urdf_path = _prepare_pybullet_urdf(urdf_path)
    try:
        p.setGravity(float(gravity[0]), float(gravity[1]), float(gravity[2]),
                     physicsClientId=client)
        flags = getattr(p, "URDF_USE_INERTIA_FROM_FILE", 0)
        body_id = p.loadURDF(
            loadable_urdf_path,
            useFixedBase=bool(fixed_base),
            flags=flags,
            physicsClientId=client,
        )

        pybullet_joint_names = []
        for joint_index in range(p.getNumJoints(body_id, physicsClientId=client)):
            joint_info = p.getJointInfo(body_id, joint_index, physicsClientId=client)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            if joint_type == p.JOINT_FIXED:
                continue
            if joint_type != p.JOINT_REVOLUTE:
                raise ValueError(
                    f"Unsupported PyBullet joint type for '{joint_name}': {joint_type}. "
                    "Only revolute/continuous joints are supported."
                )
            pybullet_joint_names.append(joint_name)
            try:
                p.changeDynamics(
                    body_id,
                    joint_index,
                    linearDamping=0.0,
                    angularDamping=0.0,
                    jointDamping=0.0,
                    physicsClientId=client,
                )
            except TypeError:
                # Older PyBullet builds may not accept the full keyword set.
                pass

        if pybullet_joint_names != joint_names:
            raise ValueError(
                "PyBullet movable joint order does not match the requested joint order. "
                f"Expected {joint_names}, got {pybullet_joint_names}."
            )

        tau = np.zeros_like(q)
        for k in range(q.shape[0]):
            tau[k] = np.asarray(
                p.calculateInverseDynamics(
                    body_id,
                    q[k].tolist(),
                    dq[k].tolist(),
                    ddq[k].tolist(),
                    physicsClientId=client,
                ),
                dtype=float,
            )
        return tau
    finally:
        p.disconnect(physicsClientId=client)
        if temp_urdf_path is not None:
            Path(temp_urdf_path).unlink(missing_ok=True)


def compute_comparison_metrics(tau_reference: np.ndarray,
                               tau_candidate: np.ndarray,
                               tolerance_abs: float,
                               tolerance_normalized_rms: float,
                               relative_epsilon: float = 1e-9) -> dict:
    """Compute torque comparison metrics and pass/fail flags."""
    tau_reference = np.asarray(tau_reference, dtype=float)
    tau_candidate = np.asarray(tau_candidate, dtype=float)

    if tau_reference.ndim != 2 or tau_candidate.shape != tau_reference.shape:
        raise ValueError("Torque arrays must both have shape (N, nDoF).")

    tau_error = tau_candidate - tau_reference
    tau_abs_error = np.abs(tau_error)
    tau_rel_error = tau_abs_error / np.maximum(np.abs(tau_reference), relative_epsilon)

    max_abs_error_per_joint = np.max(tau_abs_error, axis=0)
    rms_error_per_joint = np.sqrt(np.mean(tau_error ** 2, axis=0))
    reference_rms_per_joint = np.sqrt(np.mean(tau_reference ** 2, axis=0))
    reference_scale_per_joint = np.maximum(
        reference_rms_per_joint,
        max(float(tolerance_abs), relative_epsilon),
    )
    normalized_rms_error_per_joint = rms_error_per_joint / reference_scale_per_joint

    global_rms_reference = float(np.sqrt(np.mean(tau_reference ** 2)))
    global_rms_error = float(np.sqrt(np.mean(tau_error ** 2)))
    global_reference_scale = max(
        global_rms_reference,
        float(tolerance_abs),
        relative_epsilon,
    )
    global_normalized_rms_error = global_rms_error / global_reference_scale

    per_joint_pass = (
        (max_abs_error_per_joint <= float(tolerance_abs))
        & (normalized_rms_error_per_joint <= float(tolerance_normalized_rms))
    )
    passed = bool(np.all(per_joint_pass))

    return {
        "tau_error": tau_error,
        "tau_abs_error": tau_abs_error,
        "tau_rel_error": tau_rel_error,
        "max_abs_error_per_joint": max_abs_error_per_joint,
        "rms_error_per_joint": rms_error_per_joint,
        "reference_rms_per_joint": reference_rms_per_joint,
        "reference_scale_per_joint": reference_scale_per_joint,
        "normalized_rms_error_per_joint": normalized_rms_error_per_joint,
        "global_max_abs_error": float(np.max(tau_abs_error)),
        "global_rms_error": global_rms_error,
        "global_normalized_rms_error": float(global_normalized_rms_error),
        "per_joint_pass": per_joint_pass,
        "passed": passed,
    }


class PyBulletValidationRunner:
    """Standalone validation workflow for PyBullet torque comparison."""

    def __init__(self, config_path: str | dict):
        self.config_path = config_path
        if isinstance(config_path, dict):
            self.cfg = load_pybullet_validation_config_dict(config_path)
        else:
            self.cfg = load_pybullet_validation_config(config_path)
        self.output_dir = None
        self.logger = None

    def run(self) -> dict:
        cfg = self.cfg
        robot = parse_urdf(cfg["urdf_path"])
        output_root = Path(cfg["output_dir"])
        self.output_dir = output_root / robot.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            str(self.output_dir),
            name="pybullet_validation",
            log_filename="pybullet_validation.log",
        )
        log = self.logger

        validation_joint_names = _resolve_validation_joint_order(
            robot.revolute_joint_names,
            cfg.get("joint_name_order"),
        )
        notes = [
            "Validation uses PyBullet inverse dynamics in DIRECT mode.",
            "This workflow validates rigid-body inverse dynamics only; it does not "
            "run a closed-loop control simulation.",
            "Friction is not included in the PyBullet comparison path.",
        ]
        for wf_note in cfg.get("_workflow_notes", []):
            notes.append(wf_note)
        if validation_joint_names != robot.revolute_joint_names:
            notes.append("Joint order override applied explicitly from config.")

        log.info("=" * 60)
        log.info("PYBULLET URDF CONSISTENCY VALIDATION")
        log.info("=" * 60)
        log.info("Robot: '%s', %d DoF", robot.name, robot.nDoF)
        log.info("Validation joint order: %s", validation_joint_names)
        log.warning("Friction is not included in the PyBullet comparison path.")

        validation_gravity = np.asarray(cfg["gravity"], dtype=float)
        if not np.allclose(validation_gravity, GRAVITY, atol=1e-12, rtol=0.0):
            raise ValueError(
                "PyBullet validation received gravity "
                f"{validation_gravity.tolist()}, but the Newton-Euler regressor uses "
                f"the hardcoded math_utils.GRAVITY constant {GRAVITY.tolist()} and "
                "cannot be overridden."
            )

        t, q_parser, dq_parser, ddq_parser, sample_rate, _ = replay_excitation_trajectory(
            cfg["excitation_file"],
            cfg["base_frequency_hz"],
            cfg["trajectory_duration_periods"],
            cfg["sample_rate_hz"],
        )
        log.info("Replayed %d excitation samples at %.3f Hz", t.size, sample_rate)

        q = _reorder_columns(q_parser, robot.revolute_joint_names, validation_joint_names)
        dq = _reorder_columns(dq_parser, robot.revolute_joint_names, validation_joint_names)
        ddq = _reorder_columns(ddq_parser, robot.revolute_joint_names, validation_joint_names)

        kin = RobotKinematics(robot, log)
        if not np.allclose(kin.Tw_0, np.eye(4), atol=1e-12, rtol=0.0):
            log.info("Non-identity base transform Tw_0 detected; validation uses it as parsed.")
        tau_pipeline_parser = _compute_pipeline_torques(kin, q_parser, dq_parser, ddq_parser)
        tau_pipeline = _reorder_columns(
            tau_pipeline_parser, robot.revolute_joint_names, validation_joint_names
        )

        log.info("Computing PyBullet inverse-dynamics torques")
        tau_pybullet = compute_torques(
            cfg["urdf_path"],
            validation_joint_names,
            q,
            dq,
            ddq,
            cfg["gravity"],
            fixed_base=cfg["use_fixed_base"],
        )

        comparison = compute_comparison_metrics(
            tau_pipeline,
            tau_pybullet,
            cfg["comparison"]["tolerance_abs"],
            cfg["comparison"]["tolerance_normalized_rms"],
        )

        tol_abs = float(cfg["comparison"]["tolerance_abs"])
        for jidx, jname in enumerate(validation_joint_names):
            ref_rms = float(comparison["reference_rms_per_joint"][jidx])
            if ref_rms < tol_abs:
                notes.append(
                    f"Joint '{jname}' has near-zero reference torque "
                    f"(RMS={ref_rms:.2g}); normalized metrics may be "
                    "uninformative for this joint."
                )

        data_path = self.output_dir / "pybullet_validation_data.npz"
        np.savez(
            str(data_path),
            t=t,
            q=q,
            dq=dq,
            ddq=ddq,
            tau_pipeline=tau_pipeline,
            tau_pybullet=tau_pybullet,
            tau_error=comparison["tau_error"],
            tau_abs_error=comparison["tau_abs_error"],
            tau_rel_error=comparison["tau_rel_error"],
        )
        log.info("Validation data saved to %s", data_path)

        summary = {
            "robot_name": robot.name,
            "nDoF": robot.nDoF,
            "joint_names": validation_joint_names,
            "sample_count": int(t.size),
            "sample_rate_hz": float(sample_rate),
            "gravity": [float(x) for x in cfg["gravity"]],
            "use_fixed_base": bool(cfg["use_fixed_base"]),
            "tolerance_abs": float(cfg["comparison"]["tolerance_abs"]),
            "tolerance_normalized_rms": float(
                cfg["comparison"]["tolerance_normalized_rms"]
            ),
            "method": "newton_euler",
            "max_abs_error_per_joint": comparison["max_abs_error_per_joint"].tolist(),
            "rms_error_per_joint": comparison["rms_error_per_joint"].tolist(),
            "normalized_rms_error_per_joint": (
                comparison["normalized_rms_error_per_joint"].tolist()
            ),
            "per_joint_pass": comparison["per_joint_pass"].tolist(),
            "global_max_abs_error": comparison["global_max_abs_error"],
            "global_rms_error": comparison["global_rms_error"],
            "global_normalized_rms_error": comparison["global_normalized_rms_error"],
            "passed": comparison["passed"],
            "notes": notes,
        }
        summary_path = self.output_dir / "pybullet_validation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("Validation summary saved to %s", summary_path)

        if summary["passed"]:
            log.info("Validation PASSED")
        else:
            log.warning("Validation FAILED")
        log.info("=" * 60)
        return summary


def _compute_pipeline_torques(kin: RobotKinematics,
                              q: np.ndarray,
                              dq: np.ndarray,
                              ddq: np.ndarray) -> np.ndarray:
    """Compute reference torques from the pipeline's Newton-Euler regressor."""
    pi = kin.PI.flatten()
    tau = np.zeros((q.shape[0], kin.nDoF))
    for k in range(q.shape[0]):
        tau[k] = newton_euler_regressor(kin, q[k], dq[k], ddq[k]) @ pi
    return tau


def _resolve_validation_joint_order(parser_joint_names, override_joint_names):
    """Validate an optional explicit joint order override."""
    parser_joint_names = [str(name) for name in parser_joint_names]
    if override_joint_names is None:
        return parser_joint_names

    override_joint_names = [str(name) for name in override_joint_names]
    if len(override_joint_names) != len(parser_joint_names):
        raise ValueError(
            "joint_name_order must contain exactly the revolute/continuous joints "
            f"from the URDF. Expected {len(parser_joint_names)} names, got "
            f"{len(override_joint_names)}."
        )
    if set(override_joint_names) != set(parser_joint_names):
        raise ValueError(
            "joint_name_order must contain exactly the URDF revolute joint names. "
            f"Expected {parser_joint_names}, got {override_joint_names}."
        )
    return override_joint_names


def _reorder_columns(values: np.ndarray, source_names, target_names) -> np.ndarray:
    """Reorder a (N, nDoF) array from source_names order into target_names order."""
    source_names = [str(name) for name in source_names]
    target_names = [str(name) for name in target_names]
    indices = [source_names.index(name) for name in target_names]
    return values[:, indices]


def _load_excitation_artifact(excitation_file: str) -> dict:
    """Load and validate the excitation artifact contract."""
    required_keys = {"params", "freqs", "q0", "basis", "optimize_phase"}
    with np.load(excitation_file) as data:
        missing = required_keys.difference(data.files)
        if missing:
            raise ValueError(
                "Excitation artifact is missing required fields: "
                f"{sorted(missing)}."
            )
        return {
            "params": np.asarray(data["params"], dtype=float),
            "freqs": np.asarray(data["freqs"], dtype=float),
            "q0": np.asarray(data["q0"], dtype=float),
            "basis": str(np.asarray(data["basis"]).item()),
            "optimize_phase": bool(np.asarray(data["optimize_phase"]).item()),
        }


def _prepare_pybullet_urdf(urdf_path: str):
    """Return a PyBullet-loadable URDF path, patching missing revolute limits."""
    source_path = Path(urdf_path)
    if source_path.suffix.lower() == ".xacro":
        raise RuntimeError(
            "PyBullet validation currently requires a resolved .urdf file. "
            "Provide a plain URDF path instead of a .xacro file."
        )

    tree = ET.parse(source_path)
    root = tree.getroot()
    modified = False
    default_limit_attrs = {
        "lower": "-1e30",
        "upper": "1e30",
        "effort": "1e30",
        "velocity": "1e30",
    }

    for joint_el in root.findall("joint"):
        if joint_el.get("type") not in ("revolute", "continuous"):
            continue
        limit_el = joint_el.find("limit")
        if limit_el is None:
            limit_el = ET.SubElement(joint_el, "limit")
            modified = True
        for key, value in default_limit_attrs.items():
            if limit_el.get(key) is None:
                limit_el.set(key, value)
                modified = True

    if not modified:
        return str(source_path), None

    with tempfile.NamedTemporaryFile(
        suffix=".urdf",
        delete=False,
        mode="wb",
    ) as tmp:
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        return tmp.name, tmp.name


def _import_pybullet():
    """Import PyBullet lazily so the core pipeline remains optional."""
    try:
        return importlib.import_module("pybullet")
    except ImportError as exc:
        raise RuntimeError(
            "PyBullet is required for validation but is not installed. "
            "Install it with 'pip install pybullet' and retry."
        ) from exc


def _resolve_paths(cfg: dict, resolve_relative_to: str | Path | None) -> dict:
    """Resolve path-valued fields relative to the given config directory."""
    if resolve_relative_to is None:
        return cfg

    base_dir = Path(resolve_relative_to).resolve()
    resolved = deepcopy(cfg)
    resolved["urdf_path"] = resolve_path_value(resolved.get("urdf_path"), base_dir)
    resolved["excitation_file"] = resolve_path_value(
        resolved.get("excitation_file"),
        base_dir,
    )
    resolved["output_dir"] = resolve_path_value(resolved.get("output_dir"), base_dir)
    return resolved


def _validate_config(cfg: dict, config_path: str):
    """Validate the merged validation config."""
    if not cfg.get("urdf_path"):
        raise ValueError(f"[{config_path}] 'urdf_path' must be specified.")
    urdf_path = Path(cfg["urdf_path"])
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF/XACRO file not found: {urdf_path}")

    if not cfg.get("excitation_file"):
        raise ValueError(f"[{config_path}] 'excitation_file' must be specified.")
    excitation_file = Path(cfg["excitation_file"])
    if not excitation_file.exists():
        raise FileNotFoundError(f"Excitation artifact not found: {excitation_file}")

    if cfg["base_frequency_hz"] <= 0:
        raise ValueError("'base_frequency_hz' must be > 0.")
    if cfg["trajectory_duration_periods"] <= 0:
        raise ValueError("'trajectory_duration_periods' must be > 0.")
    if cfg["sample_rate_hz"] < 0:
        raise ValueError("'sample_rate_hz' must be >= 0. Use 0 for auto.")

    gravity = cfg["gravity"]
    if not isinstance(gravity, list) or len(gravity) != 3:
        raise ValueError("'gravity' must be a 3-element list.")

    if cfg.get("joint_name_order") is not None:
        if not isinstance(cfg["joint_name_order"], list):
            raise ValueError("'joint_name_order' must be null or a list of joint names.")
        if not all(isinstance(name, str) and name for name in cfg["joint_name_order"]):
            raise ValueError("'joint_name_order' entries must be non-empty strings.")

    comparison = cfg["comparison"]
    if comparison["tolerance_abs"] < 0:
        raise ValueError("'comparison.tolerance_abs' must be >= 0.")
    if comparison["tolerance_normalized_rms"] < 0:
        raise ValueError("'comparison.tolerance_normalized_rms' must be >= 0.")
