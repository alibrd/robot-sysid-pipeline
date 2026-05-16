"""Load and validate the JSON configuration file."""
import json
from copy import deepcopy
from pathlib import Path

from .config_utils import deep_merge, resolve_path_value
from .torque_constraints import VALID_ENVELOPE_TYPES, VALID_TORQUE_METHODS

_VALID_METHODS = {"newton_euler", "euler_lagrange"}
_VALID_BASIS = {"cosine", "sine", "both"}
_VALID_FRICTION = {"none", "viscous", "coulomb", "viscous_coulomb"}
_VALID_SOLVERS = {"ols", "wls", "bounded_ls"}
_VALID_FEASIBILITY = {"none", "lmi", "cholesky"}


def load_default_config() -> dict:
    """Load the repository default pipeline config."""
    default_path = Path(__file__).resolve().parent.parent / "config" / "default_config.json"
    with open(default_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_config(config_path: str) -> dict:
    """Load a pipeline config file, merge defaults, resolve paths, and validate."""
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8-sig") as f:
        user_cfg = json.load(f)

    return load_config_dict(
        user_cfg,
        config_path=str(config_file),
        resolve_relative_to=config_file.parent,
        validate=True,
    )


def load_config_dict(user_cfg: dict,
                     config_path: str = "<inline>",
                     resolve_relative_to: str | Path | None = None,
                     validate: bool = True) -> dict:
    """Merge a pipeline config dict with defaults, resolve paths, and validate.

    Warning: when *resolve_relative_to* is None (default), relative paths in
    the config dict are **not** resolved and may cause FileNotFoundError at
    runtime.  Pass the directory containing the config file to enable path
    resolution.
    """
    defaults = load_default_config()
    user_cfg = _normalize_pipeline_mode_fields(user_cfg)
    cfg = deep_merge(defaults, user_cfg)
    cfg = _strip_deprecated_excitation_keys(cfg)
    cfg = _resolve_paths(cfg, resolve_relative_to)
    if validate:
        _validate(cfg, config_path)
    return cfg


def _resolve_paths(cfg: dict, resolve_relative_to: str | Path | None) -> dict:
    """Resolve path-valued fields relative to the given config directory."""
    if resolve_relative_to is None:
        return cfg

    base_dir = Path(resolve_relative_to).resolve()
    resolved = deepcopy(cfg)
    resolved["urdf_path"] = resolve_path_value(resolved.get("urdf_path"), base_dir)
    resolved["output_dir"] = resolve_path_value(resolved.get("output_dir"), base_dir)

    resolved["checkpoint_dir"] = resolve_path_value(
        resolved.get("checkpoint_dir"), base_dir
    )

    identification = resolved.get("identification", {})
    identification["data_file"] = resolve_path_value(
        identification.get("data_file"),
        base_dir,
    )
    cache_cfg = identification.get("observation_matrix_cache")
    if isinstance(cache_cfg, dict):
        cache_cfg = deepcopy(cache_cfg)
        cache_cfg["load_from"] = resolve_path_value(
            cache_cfg.get("load_from"),
            base_dir,
        )
        identification["observation_matrix_cache"] = cache_cfg
    resolved["identification"] = identification
    return resolved


def _normalize_pipeline_mode_fields(user_cfg: dict) -> dict:
    """Accept workflow-style nested pipeline mode fields in pipeline configs."""
    normalized = deepcopy(user_cfg)
    pipeline_section = normalized.get("pipeline")
    if not isinstance(pipeline_section, dict):
        return normalized

    for key in ("excitation_only", "checkpoint_dir"):
        if key not in normalized and key in pipeline_section:
            normalized[key] = pipeline_section[key]
    return normalized


def _strip_deprecated_excitation_keys(cfg: dict) -> dict:
    """Remove excitation config keys kept only for legacy compatibility."""
    cleaned = deepcopy(cfg)
    excitation = deepcopy(cleaned.get("excitation", {}))
    excitation.pop("constraint_style", None)
    excitation.pop("optimizer_pop_size", None)
    cleaned["excitation"] = excitation
    return cleaned


def _validate(cfg: dict, path: str):
    if not cfg.get("urdf_path"):
        raise ValueError(f"[{path}] 'urdf_path' must be specified.")
    urdf = Path(cfg["urdf_path"])
    if not urdf.exists():
        raise FileNotFoundError(f"URDF/XACRO file not found: {urdf}")

    if cfg["method"] not in _VALID_METHODS:
        raise ValueError(f"'method' must be one of {_VALID_METHODS}, got '{cfg['method']}'")

    exc = cfg["excitation"]
    if exc["basis_functions"] not in _VALID_BASIS:
        raise ValueError(f"'basis_functions' must be one of {_VALID_BASIS}")
    if exc["num_harmonics"] < 1:
        raise ValueError("'num_harmonics' must be >= 1")
    if exc["base_frequency_hz"] <= 0:
        raise ValueError("'base_frequency_hz' must be > 0")

    # Sine-only endpoint guarantee requires integer trajectory_duration_periods
    n_periods = exc.get("trajectory_duration_periods", 1)
    if exc["basis_functions"] == "sine" and n_periods != int(n_periods):
        raise ValueError(
            f"'trajectory_duration_periods' must be an integer for sine-only "
            f"basis (got {n_periods}). Sine dq(T)=0 is only guaranteed when T "
            f"is an integer multiple of the base period 1/f0."
        )

    torque_method = exc.get("torque_constraint_method", "none")
    if torque_method not in VALID_TORQUE_METHODS:
        raise ValueError(
            f"'excitation.torque_constraint_method' must be one of "
            f"{sorted(VALID_TORQUE_METHODS)}, got '{torque_method}'"
        )

    oversample = exc.get("torque_validation_oversample_factor", 1)
    if oversample < 1:
        raise ValueError("'excitation.torque_validation_oversample_factor' must be >= 1")

    torque_cfg = exc.get("torque_constraint", {})
    if torque_method == "robust_box":
        if torque_cfg.get("relative_uncertainty") is None:
            raise ValueError(
                "robust_box requires excitation.torque_constraint.relative_uncertainty"
            )
        if float(torque_cfg.get("absolute_uncertainty_floor", 0.0)) < 0:
            raise ValueError(
                "'excitation.torque_constraint.absolute_uncertainty_floor' must be >= 0"
            )
    if torque_method == "chance":
        if torque_cfg.get("relative_stddev") is None:
            raise ValueError(
                "chance requires excitation.torque_constraint.relative_stddev"
            )
        confidence = float(torque_cfg.get("chance_confidence", 0.0))
        if not (0.5 < confidence < 1.0):
            raise ValueError(
                "'excitation.torque_constraint.chance_confidence' must lie in (0.5, 1.0)"
            )
        if float(torque_cfg.get("absolute_stddev_floor", 0.0)) < 0:
            raise ValueError(
                "'excitation.torque_constraint.absolute_stddev_floor' must be >= 0"
            )
    if torque_method == "actuator_envelope":
        env_type = torque_cfg.get("envelope_type")
        if env_type not in VALID_ENVELOPE_TYPES:
            raise ValueError(
                f"'excitation.torque_constraint.envelope_type' must be one of "
                f"{sorted(VALID_ENVELOPE_TYPES)}"
            )
        if env_type == "speed_linear":
            if float(torque_cfg.get("velocity_reference", 0.0)) <= 0.0:
                raise ValueError(
                    "'excitation.torque_constraint.velocity_reference' must be > 0 "
                    "for the speed_linear actuator envelope."
                )
            if float(torque_cfg.get("min_scale", 0.0)) <= 0.0:
                raise ValueError(
                    "'excitation.torque_constraint.min_scale' must be > 0"
                )
            if float(torque_cfg.get("max_scale", 0.0)) <= 0.0:
                raise ValueError(
                    "'excitation.torque_constraint.max_scale' must be > 0"
                )
    if torque_method == "sequential_redesign":
        if cfg["method"] != "newton_euler":
            raise ValueError(
                "sequential_redesign is only supported with method='newton_euler' in v1."
            )
        if cfg.get("identification", {}).get("data_file") is not None:
            raise ValueError(
                "sequential_redesign is supported only for synthetic-data runs "
                "(identification.data_file must be null)."
            )
        if int(torque_cfg.get("max_iterations", 0)) < 1:
            raise ValueError(
                "'excitation.torque_constraint.max_iterations' must be >= 1"
            )
        if float(torque_cfg.get("convergence_tol", 0.0)) < 0.0:
            raise ValueError(
                "'excitation.torque_constraint.convergence_tol' must be >= 0"
            )

    if cfg["friction"]["model"] not in _VALID_FRICTION:
        raise ValueError(f"'friction.model' must be one of {_VALID_FRICTION}")

    ident = cfg["identification"]
    if ident["solver"] not in _VALID_SOLVERS:
        raise ValueError(f"'solver' must be one of {_VALID_SOLVERS}")
    if ident["feasibility_method"] not in _VALID_FEASIBILITY:
        raise ValueError(f"'feasibility_method' must be one of {_VALID_FEASIBILITY}")

    # Pipeline partitioning: excitation_only and checkpoint_dir are mutually
    # exclusive run modes.
    if cfg.get("excitation_only") and cfg.get("checkpoint_dir"):
        raise ValueError(
            "'excitation_only' and 'checkpoint_dir' are mutually exclusive. "
            "Set excitation_only=true to stop after Stage 6, or set "
            "checkpoint_dir to resume from a previous run, but not both."
        )

    # Adapted-URDF export (Stage 12) — minimal foot-gun checks on filenames.
    exp = cfg.get("export") or {}
    if exp.get("enabled", False):
        for fn_key in ("urdf_filename", "friction_sidecar_filename"):
            v = exp.get(fn_key)
            if v is None:
                continue
            if not isinstance(v, str):
                raise ValueError(
                    f"[{path}] 'export.{fn_key}' must be a string, got {type(v).__name__}"
                )
            v_path = Path(v)
            if v_path.is_absolute() or ".." in v_path.parts:
                raise ValueError(
                    f"[{path}] 'export.{fn_key}' must be a plain filename "
                    f"relative to output_dir, got '{v}'"
                )
