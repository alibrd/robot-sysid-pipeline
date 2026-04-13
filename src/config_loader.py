"""Load and validate the JSON configuration file."""
import json
import warnings
from pathlib import Path

_VALID_METHODS = {"newton_euler", "euler_lagrange"}
_VALID_BASIS = {"cosine", "sine", "both"}
_VALID_CONSTRAINT_STYLES = {"legacy_excTrajGen", "urdf_reference", "literature_standard"}
_VALID_FRICTION = {"none", "viscous", "coulomb", "viscous_coulomb"}
_VALID_SOLVERS = {"ols", "wls", "bounded_ls"}
_VALID_FEASIBILITY = {"none", "lmi", "cholesky"}


def load_config(config_path: str) -> dict:
    """Load user JSON config, merge with defaults, and validate."""
    default_path = Path(__file__).resolve().parent.parent / "config" / "default_config.json"
    with open(default_path, "r") as f:
        defaults = json.load(f)

    with open(config_path, "r") as f:
        user_cfg = json.load(f)

    cfg = _deep_merge(defaults, user_cfg)
    _validate(cfg, config_path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


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
