"""Load and validate the JSON configuration file."""
import json
import warnings
from copy import deepcopy
from pathlib import Path

from .config_utils import deep_merge, resolve_path_value

_VALID_METHODS = {"newton_euler", "euler_lagrange"}
_VALID_BASIS = {"cosine", "sine", "both"}
_VALID_CONSTRAINT_STYLES = {"legacy_excTrajGen", "urdf_reference", "literature_standard"}
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
    cfg = deep_merge(defaults, deepcopy(user_cfg))
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

    identification = resolved.get("identification", {})
    identification["data_file"] = resolve_path_value(
        identification.get("data_file"),
        base_dir,
    )
    resolved["identification"] = identification
    return resolved


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
    if exc["constraint_style"] not in _VALID_CONSTRAINT_STYLES:
        raise ValueError(f"'constraint_style' must be one of {_VALID_CONSTRAINT_STYLES}")
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

    if cfg["friction"]["model"] not in _VALID_FRICTION:
        raise ValueError(f"'friction.model' must be one of {_VALID_FRICTION}")

    ident = cfg["identification"]
    if ident["solver"] not in _VALID_SOLVERS:
        raise ValueError(f"'solver' must be one of {_VALID_SOLVERS}")
    if ident["feasibility_method"] not in _VALID_FEASIBILITY:
        raise ValueError(f"'feasibility_method' must be one of {_VALID_FEASIBILITY}")

    # Normalize: "cholesky" is a deprecated alias for "lmi" (same implementation)
    if ident["feasibility_method"] == "cholesky":
        warnings.warn(
            "feasibility_method='cholesky' is a deprecated alias for 'lmi'. "
            "No separate Cholesky-factored reparameterisation is implemented; "
            "both use eigenvalue-clipping projection of the pseudo-inertia "
            "matrix. Use 'lmi' directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        ident["feasibility_method"] = "lmi"

    # Constrained identification requires full 10-per-link parameter blocks,
    # which are only available with the newton_euler regressor.  The EL
    # regressor drops zero columns, producing a reduced vector that cannot
    # be mapped back to per-link pseudo-inertia constraints.
    if cfg["method"] == "euler_lagrange" and ident["feasibility_method"] != "none":
        raise ValueError(
            "Constrained identification (feasibility_method='lmi') is not "
            "supported with the euler_lagrange method. The EL regressor "
            "produces a reduced parameter vector that cannot be mapped to "
            "per-link pseudo-inertia constraints. Use method='newton_euler' "
            "for constrained identification, or set feasibility_method='none'."
        )
