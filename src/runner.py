"""Unified single-config runner for the system identification toolchain.

Loads one JSON config, deep-merges it with `config/default_config.json`,
resolves relative paths once, validates the unified-schema fields, and then
delegates each enabled stage to the existing internal classes/functions:

    stages.excitation / stages.identification  -> SystemIdentificationPipeline
    stages.validation (validation.source=="pybullet")    -> PyBulletValidationRunner
    stages.validation (validation.source==<path>)        -> MeasurementValidationRunner
    stages.report                              -> export_validation_report
    stages.benchmark                           -> export_validation_benchmark
    stages.plot                                -> src.plot_runner

The runner is a translation layer; it does not modify any scientific code.

Mode 1 vs Mode 2 is read directly off the config:
    Mode 1  <=>  identification.source == "excitation"
    Mode 2  <=>  identification.source is a path to a pipeline-compatible
                 measurement .npz (keys q, dq, ddq, tau, optional fs).

The validation source is orthogonal: validation.source can be "pybullet" or
a path to a measurement file in either mode.
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

from .config_loader import load_config_dict, load_default_config
from .config_utils import deep_merge, resolve_path_value
from .measurement_validation import MeasurementValidationRunner
from .pipeline import SystemIdentificationPipeline
from .pybullet_validation import PyBulletValidationRunner
from .pybullet_validation_benchmark import export_validation_benchmark
from .pybullet_validation_report import export_validation_report
from .trajectory import build_frequencies


_VALID_STAGES = (
    "excitation",
    "identification",
    "validation",
    "report",
    "benchmark",
    "plot",
)

_KNOWN_TOP_LEVEL_KEYS = {
    "urdf_path",
    "output_dir",
    "method",
    "stages",
    "checkpoint",
    "identification",
    "validation",
    "joint_limits",
    "excitation",
    "friction",
    "filtering",
    "downsampling",
    "plot",
    "export",
    "advanced",
}

_REMOVED_TOP_LEVEL_KEYS = {
    "resume": (
        "Top-level 'resume' block was removed. Use the flat top-level "
        "'checkpoint' key instead (set to a path, or null)."
    ),
    "validation_pybullet": (
        "Top-level 'validation_pybullet' block was renamed to 'validation'. "
        "Add 'source: \"pybullet\"' (or a path to a measurements .npz) inside it."
    ),
    "report": (
        "Empty 'report' block was removed; the stage flag 'stages.report' "
        "already toggles behaviour."
    ),
    "benchmark": (
        "Empty 'benchmark' block was removed; the stage flag "
        "'stages.benchmark' already toggles behaviour."
    ),
}

_REMOVED_STAGE_FLAGS = {
    "validation_pybullet": (
        "Stage flag 'stages.validation_pybullet' was renamed to "
        "'stages.validation'. Validation backend is selected via "
        "'validation.source' (\"pybullet\" or a measurements path)."
    ),
}

_REMOVED_IDENTIFICATION_KEYS = {
    "data_file": (
        "'identification.data_file' was replaced by 'identification.source'. "
        "Use source=\"excitation\" for synthetic data, or source=\"<path>\" "
        "for real measurements."
    ),
}

_PIPELINE_SUBDIR = "pipeline"
_VALIDATION_SUBDIR = "validation"
_PLOT_SUBDIR = "plots"
_RUNNER_ONLY_KEYS = frozenset({
    "stages",
    "checkpoint",
    "validation",
    "plot",
    "report",
    "benchmark",
})

_PATH_KEYS_TOP_LEVEL = ("urdf_path", "output_dir")


class UnifiedRunner:
    """Coordinate excitation / identification / validation / report / benchmark / plot stages."""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).resolve())
        self.cfg = self._load_unified_config(self.config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_only(self, stage: str) -> None:
        """Disable every stage except *stage*."""
        self._require_known_stage(stage)
        self.cfg["stages"] = {name: (name == stage) for name in _VALID_STAGES}

    def disable_stage(self, stage: str) -> None:
        """Force-disable a single stage."""
        self._require_known_stage(stage)
        self.cfg["stages"][stage] = False

    def set_resume(self, checkpoint_dir: str) -> None:
        """Override top-level 'checkpoint' with a path resolved against the caller's cwd.

        Note: JSON 'checkpoint' is resolved against the config file's
        directory; this method resolves against Path.cwd() instead, matching
        standard CLI path semantics.
        """
        if checkpoint_dir in (None, ""):
            self.cfg["checkpoint"] = None
            return
        path = Path(checkpoint_dir)
        if not path.is_absolute():
            path = Path.cwd() / path
        self.cfg["checkpoint"] = str(path.resolve())

    def print_resolved_config(self) -> None:
        """Print the merged + resolved config (for --dry-run)."""
        cfg_to_print = deepcopy(self.cfg)
        cfg_to_print.pop("_config_path", None)
        cfg_to_print.pop("_config_dir", None)
        print(json.dumps(cfg_to_print, indent=2))

    def run(self) -> int:
        """Execute the configured stages. Returns 0 on success, 1 on validation failure."""
        ctx = self._validate_and_prepare()

        pipeline_runner = None
        if ctx["run_pipeline"]:
            pipeline_runner = SystemIdentificationPipeline(ctx["pipeline_cfg"])
            pipeline_runner.run()

        validation_summary = None
        validation_runner = None
        if ctx["run_validation"]:
            if ctx["validation_backend"] == "pybullet":
                excitation_path = Path(ctx["validation_cfg"]["excitation_file"])
                _check_excitation_artifact_frequencies(
                    excitation_path,
                    ctx["validation_cfg"]["base_frequency_hz"],
                )
                validation_runner = PyBulletValidationRunner(ctx["validation_cfg"])
                validation_summary = validation_runner.run()
            else:
                validation_runner = MeasurementValidationRunner(
                    ctx["validation_cfg"]
                )
                validation_summary = validation_runner.run()

        if ctx["run_report"]:
            target_dir = (
                str(validation_runner.output_dir)
                if validation_runner is not None
                else ctx["report_validation_dir"]
            )
            export_validation_report(target_dir)

        if ctx["run_benchmark"]:
            root = ctx["benchmark_validation_root"]
            export_validation_benchmark(root, root)

        if ctx["run_plot"]:
            from .plot_runner import plot_excitation_outputs
            plot_excitation_outputs(
                pipeline_dir=Path(self.cfg["output_dir"]) / _PIPELINE_SUBDIR,
                output_dir=Path(self.cfg["output_dir"]) / _PLOT_SUBDIR,
                plot_cfg=self.cfg.get("plot", {}),
            )

        if validation_summary is not None and not validation_summary.get("passed", True):
            return 1
        return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_unified_config(self, config_path: str) -> dict:
        config_file = Path(config_path).resolve()
        with open(config_file, "r", encoding="utf-8-sig") as f:
            user_cfg = json.load(f)
        cfg = deep_merge(load_default_config(), user_cfg)
        cfg = self._resolve_paths(cfg, config_file.parent)
        cfg["_config_path"] = str(config_file)
        cfg["_config_dir"] = str(config_file.parent)
        return cfg

    @staticmethod
    def _resolve_paths(cfg: dict, base_dir: Path) -> dict:
        resolved = deepcopy(cfg)
        for key in _PATH_KEYS_TOP_LEVEL:
            resolved[key] = resolve_path_value(resolved.get(key), base_dir)

        if resolved.get("checkpoint"):
            resolved["checkpoint"] = resolve_path_value(
                resolved.get("checkpoint"), base_dir
            )

        identification = resolved.get("identification", {}) or {}
        source = identification.get("source")
        if isinstance(source, str) and source not in ("excitation", ""):
            identification["source"] = resolve_path_value(source, base_dir)
        resolved["identification"] = identification

        validation = resolved.get("validation", {}) or {}
        val_source = validation.get("source")
        if isinstance(val_source, str) and val_source not in ("pybullet", ""):
            validation["source"] = resolve_path_value(val_source, base_dir)
        resolved["validation"] = validation

        advanced = resolved.get("advanced", {}) or {}
        cache_cfg = advanced.get("observation_matrix_cache")
        if isinstance(cache_cfg, dict):
            cache_cfg = deepcopy(cache_cfg)
            cache_cfg["load_from"] = resolve_path_value(
                cache_cfg.get("load_from"), base_dir
            )
            advanced["observation_matrix_cache"] = cache_cfg
            resolved["advanced"] = advanced
        return resolved

    @staticmethod
    def _require_known_stage(stage: str) -> None:
        if stage not in _VALID_STAGES:
            raise ValueError(
                f"Unknown stage '{stage}'. Valid stages are {sorted(_VALID_STAGES)}."
            )

    def _validate_and_prepare(self) -> dict:
        cfg = self.cfg

        # Reject removed top-level keys with a clear migration message before
        # the generic "unknown key" rejection so the user sees the right hint.
        for removed_key, message in _REMOVED_TOP_LEVEL_KEYS.items():
            if removed_key in cfg:
                raise ValueError(f"{message}")

        unknown = set(cfg.keys()) - _KNOWN_TOP_LEVEL_KEYS - {"_config_path", "_config_dir"}
        if unknown:
            raise ValueError(
                f"Unknown top-level keys in unified config: {sorted(unknown)}. "
                f"Known keys: {sorted(_KNOWN_TOP_LEVEL_KEYS)}."
            )

        if not cfg.get("urdf_path"):
            raise ValueError("Unified config must define 'urdf_path'.")
        urdf_path = Path(cfg["urdf_path"])
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF/XACRO file not found: {urdf_path}")

        stages = cfg.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("'stages' must be a dictionary.")
        for removed_flag, message in _REMOVED_STAGE_FLAGS.items():
            if removed_flag in stages:
                raise ValueError(message)
        unknown_stages = set(stages.keys()) - set(_VALID_STAGES)
        if unknown_stages:
            raise ValueError(
                f"Unknown stage flags: {sorted(unknown_stages)}. "
                f"Valid stage flags: {sorted(_VALID_STAGES)}."
            )

        identification_block = cfg.get("identification") or {}
        for removed_key, message in _REMOVED_IDENTIFICATION_KEYS.items():
            if removed_key in identification_block:
                raise ValueError(message)

        # Source selectors. "excitation" / "pybullet" are sentinel literals;
        # anything else is interpreted as a filesystem path.
        identification_source = identification_block.get("source", "excitation")
        validation_block = cfg.get("validation") or {}
        validation_source = validation_block.get("source", "pybullet")

        identification_from_measurements = (
            isinstance(identification_source, str)
            and identification_source != "excitation"
        )
        validation_from_measurements = (
            isinstance(validation_source, str)
            and validation_source != "pybullet"
        )

        if identification_from_measurements:
            id_path = _resolve_measurement_path(
                identification_source, label="identification.source"
            )
        else:
            id_path = None
        if validation_from_measurements:
            val_path = _resolve_measurement_path(
                validation_source, label="validation.source"
            )
        else:
            val_path = None

        run_excitation = bool(stages.get("excitation", False))
        run_identification = bool(stages.get("identification", False))
        run_validation = bool(stages.get("validation", False))
        run_report = bool(stages.get("report", False))
        run_benchmark = bool(stages.get("benchmark", False))
        run_plot = bool(stages.get("plot", False))

        checkpoint_path = cfg.get("checkpoint")

        # When identifying from a measurement file there is no excitation
        # trajectory to design, so ignore the excitation stage if defaults left
        # it enabled.
        if identification_from_measurements and run_excitation:
            warnings.warn(
                "stages.excitation=true is ignored because identification.source "
                "points at a measurement file. No excitation trajectory is designed "
                "for measurement-source identification.",
                UserWarning,
                stacklevel=2,
            )
            run_excitation = False
            cfg["stages"]["excitation"] = False

        # Cross-stage validity rules.
        if checkpoint_path and run_excitation:
            raise ValueError(
                "'checkpoint' is incompatible with stages.excitation=true. "
                "A resume run reuses a saved excitation checkpoint and cannot design a "
                "new excitation in the same invocation. Either clear 'checkpoint' "
                "for a full fresh run, or set stages.excitation=false to run only "
                "identification/validation from the saved checkpoint."
            )

        if checkpoint_path and not (run_identification or run_validation):
            raise ValueError(
                "'checkpoint' is set but no enabled stage can consume it. "
                "Enable stages.identification or stages.validation."
            )

        if (
            run_identification
            and not identification_from_measurements
            and not (run_excitation or checkpoint_path)
        ):
            raise ValueError(
                "stages.identification=true with identification.source='excitation' "
                "and stages.excitation=false requires 'checkpoint'. Enable "
                "stages.excitation for a full run or set 'checkpoint' to identify "
                "from a saved excitation."
            )

        run_pipeline = run_excitation or run_identification

        validation_backend = "pybullet" if not validation_from_measurements else "measurements"

        if run_validation:
            if validation_backend == "pybullet":
                if identification_from_measurements:
                    raise ValueError(
                        "validation.source='pybullet' is not supported when "
                        "identification.source is a measurement file. PyBullet "
                        "validation replays a designed excitation trajectory; with "
                        "no excitation in this run there is no trajectory to replay. "
                        "Either set validation.source to a measurement path, or "
                        "disable validation for this Mode-2 run."
                    )
                if not (run_excitation or checkpoint_path):
                    raise ValueError(
                        "stages.validation=true with validation.source='pybullet' "
                        "requires either stages.excitation=true (to produce a fresh "
                        "excitation artifact) or 'checkpoint' to be set (to reuse a "
                        "previous artifact)."
                    )
                if not _is_module_available("pybullet"):
                    raise RuntimeError(
                        "PyBullet is required for validation.source='pybullet' "
                        "but is not installed."
                    )
            else:
                if not run_identification and not _existing_identification_results(cfg):
                    raise ValueError(
                        "stages.validation=true with validation.source=<path> "
                        "requires identified parameters. Enable stages.identification "
                        "in this run or ensure that "
                        f"{Path(cfg['output_dir']) / _PIPELINE_SUBDIR / 'identification_results.npz'} "
                        "already exists from a previous run."
                    )

        report_validation_dir = None
        if run_report:
            if run_validation:
                report_validation_dir = None
            else:
                report_validation_dir = self._existing_validation_run_dir()
                if report_validation_dir is None:
                    raise ValueError(
                        "stages.report=true requires either stages.validation=true "
                        "in the same run, or an existing validation directory under "
                        f"{Path(cfg['output_dir']) / _VALIDATION_SUBDIR}."
                    )

        benchmark_root = None
        if run_benchmark:
            benchmark_root = Path(cfg["output_dir"]) / _VALIDATION_SUBDIR
            if not benchmark_root.exists() and not run_validation:
                raise ValueError(
                    "stages.benchmark=true requires an existing validation directory at "
                    f"{benchmark_root}, or stages.validation=true in the same run."
                )
            benchmark_root = str(benchmark_root)

        if run_plot and not _is_module_available("matplotlib"):
            raise RuntimeError(
                "matplotlib is required for the plot stage but is not installed."
            )

        pipeline_cfg = None
        if run_pipeline:
            pipeline_cfg = self._pipeline_cfg_dict(
                run_excitation=run_excitation,
                run_identification=run_identification,
                checkpoint_path=checkpoint_path,
                identification_data_file=id_path,
            )

        validation_cfg = None
        if run_validation:
            if validation_backend == "pybullet":
                validation_cfg = self._pybullet_validation_cfg_dict(
                    run_excitation=run_excitation,
                    checkpoint_path=checkpoint_path,
                )
                friction_model = cfg.get("friction", {}).get("model", "none")
                if friction_model != "none":
                    warnings.warn(
                        "validation.source='pybullet' is deriving a PyBullet validation run "
                        f"from a unified config with friction.model='{friction_model}', "
                        "but the PyBullet comparison excludes friction and only validates "
                        "rigid-body inverse dynamics.",
                        UserWarning,
                        stacklevel=2,
                    )
                    validation_cfg.setdefault("_workflow_notes", []).append(
                        f"Pipeline uses friction.model='{friction_model}', but the "
                        "PyBullet comparison excludes friction. The friction component "
                        "of the identified model is not validated by this run."
                    )
            else:
                validation_cfg = self._measurement_validation_cfg_dict(
                    measurements_path=val_path,
                )

        return {
            "run_pipeline": run_pipeline,
            "run_validation": run_validation,
            "run_report": run_report,
            "run_benchmark": run_benchmark,
            "run_plot": run_plot,
            "validation_backend": validation_backend,
            "pipeline_cfg": pipeline_cfg,
            "validation_cfg": validation_cfg,
            "report_validation_dir": report_validation_dir,
            "benchmark_validation_root": benchmark_root,
        }

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    def _pipeline_cfg_dict(self, *,
                           run_excitation: bool,
                           run_identification: bool,
                           checkpoint_path: str | None,
                           identification_data_file: str | None) -> dict:
        """Translate the unified config into a SystemIdentificationPipeline dict."""
        cfg = self.cfg
        pipeline_output_dir = str(Path(cfg["output_dir"]) / _PIPELINE_SUBDIR)

        checkpoint_dir = None
        if checkpoint_path:
            cp = Path(checkpoint_path)
            # If the user pointed at a previous unified output_dir, the checkpoint
            # actually lives inside its pipeline subdirectory. If they already
            # pointed at the pipeline subdir, leave it alone.
            if cp.name == _PIPELINE_SUBDIR or (cp / "checkpoint.npz").exists():
                checkpoint_dir = str(cp)
            else:
                checkpoint_dir = str(cp / _PIPELINE_SUBDIR)

        identification_block = deepcopy(cfg.get("identification", {}))
        # Translate the user-facing `source` selector into the internal
        # `data_file` slot consumed by SystemIdentificationPipeline. The
        # scientific core never sees `source`.
        identification_block.pop("source", None)
        identification_block["data_file"] = identification_data_file

        # The user-facing `advanced.observation_matrix_cache` block is the same
        # data the internal pipeline expects under
        # identification.observation_matrix_cache.
        advanced = cfg.get("advanced") or {}
        cache_cfg = advanced.get("observation_matrix_cache")
        if isinstance(cache_cfg, dict):
            identification_block.setdefault("observation_matrix_cache", deepcopy(cache_cfg))

        pipeline_cfg = {
            "urdf_path": cfg["urdf_path"],
            "output_dir": pipeline_output_dir,
            "method": cfg.get("method", "newton_euler"),
            "excitation_only": run_excitation and not run_identification,
            "checkpoint_dir": checkpoint_dir,
            "joint_limits": deepcopy(cfg.get("joint_limits", {})),
            "excitation": deepcopy(cfg.get("excitation", {})),
            "friction": deepcopy(cfg.get("friction", {"model": "none"})),
            "identification": identification_block,
            "filtering": deepcopy(cfg.get("filtering", {})),
            "downsampling": deepcopy(cfg.get("downsampling", {})),
            "export": deepcopy(cfg.get("export", {})),
        }

        # Run the pipeline-config validator so that every existing
        # validation rule still fires (e.g. mutually-exclusive run modes and
        # torque-method preconditions).
        validated = load_config_dict(
            pipeline_cfg,
            config_path=self.config_path,
            resolve_relative_to=None,
            validate=True,
        )
        for _key in _RUNNER_ONLY_KEYS:
            validated.pop(_key, None)
        return validated

    def _pybullet_validation_cfg_dict(self, *,
                                      run_excitation: bool,
                                      checkpoint_path: str | None) -> dict:
        """Translate the unified config into a PyBulletValidationRunner dict."""
        cfg = self.cfg
        validation_output_dir = str(Path(cfg["output_dir"]) / _VALIDATION_SUBDIR)

        excitation_section = deepcopy(cfg.get("excitation", {}))
        base_frequency_hz = float(excitation_section.get("base_frequency_hz", 0.2))
        trajectory_duration_periods = float(
            excitation_section.get("trajectory_duration_periods", 1)
        )

        if run_excitation:
            excitation_file = str(
                Path(cfg["output_dir"]) / _PIPELINE_SUBDIR / "excitation_trajectory.npz"
            )
        else:
            checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_path)
            excitation_file = str(checkpoint_dir / "excitation_trajectory.npz")
            if not Path(excitation_file).exists():
                raise FileNotFoundError(
                    "Validation requires an existing excitation artifact when "
                    f"stages.excitation=false. Missing file: {excitation_file}"
                )
            meta = _checkpoint_excitation_metadata(checkpoint_dir, excitation_section)
            base_frequency_hz = meta["base_frequency_hz"]
            trajectory_duration_periods = meta["trajectory_duration_periods"]

        validation_section = deepcopy(cfg.get("validation", {}))

        return {
            "urdf_path": cfg["urdf_path"],
            "excitation_file": excitation_file,
            "output_dir": validation_output_dir,
            "base_frequency_hz": base_frequency_hz,
            "trajectory_duration_periods": trajectory_duration_periods,
            "sample_rate_hz": validation_section.get("sample_rate_hz", 0),
            "gravity": validation_section.get("gravity", [0.0, 0.0, -9.80665]),
            "use_fixed_base": validation_section.get("use_fixed_base", True),
            "joint_name_order": validation_section.get("joint_name_order", None),
            "comparison": deepcopy(
                validation_section.get(
                    "comparison",
                    {"tolerance_abs": 1e-3, "tolerance_normalized_rms": 1e-3},
                )
            ),
        }

    def _measurement_validation_cfg_dict(self, *, measurements_path: str) -> dict:
        """Translate the unified config into a MeasurementValidationRunner dict."""
        cfg = self.cfg
        validation_output_dir = str(Path(cfg["output_dir"]) / _VALIDATION_SUBDIR)
        pipeline_dir = str(Path(cfg["output_dir"]) / _PIPELINE_SUBDIR)
        return {
            "urdf_path": cfg["urdf_path"],
            "measurements_path": measurements_path,
            "pipeline_dir": pipeline_dir,
            "output_dir": validation_output_dir,
            "method": cfg.get("method", "newton_euler"),
            "friction_model": cfg.get("friction", {}).get("model", "none"),
        }

    def _resolve_checkpoint_dir(self, checkpoint_path: str | None) -> Path:
        """Return the absolute path of the pipeline subdir inside a resume target."""
        if not checkpoint_path:
            raise ValueError("'checkpoint' is required to resolve checkpoint_dir.")
        cp = Path(checkpoint_path)
        if cp.name == _PIPELINE_SUBDIR:
            return cp
        if (cp / "checkpoint.npz").exists() or (cp / "excitation_trajectory.npz").exists():
            return cp
        return cp / _PIPELINE_SUBDIR

    def _existing_validation_run_dir(self) -> str | None:
        """Find an existing flat validation directory containing a summary."""
        validation_root = Path(self.cfg["output_dir"]) / _VALIDATION_SUBDIR
        if (validation_root / "pybullet_validation_summary.json").exists():
            return str(validation_root)
        if (validation_root / "measurement_validation_summary.json").exists():
            return str(validation_root)
        return None


# ----------------------------------------------------------------------
# Pure helper utilities; these contain no scientific code.
# ----------------------------------------------------------------------


def _checkpoint_excitation_metadata(checkpoint_dir: Path, fallback_excitation: dict) -> dict:
    """Return excitation metadata matching a saved checkpoint, with fallbacks."""
    excitation = deepcopy(fallback_excitation)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_config_path = checkpoint_dir / "checkpoint_config.json"
    if checkpoint_config_path.exists():
        with open(checkpoint_config_path, "r", encoding="utf-8-sig") as f:
            checkpoint_cfg = json.load(f)
        checkpoint_excitation = checkpoint_cfg.get("excitation", {})
        if isinstance(checkpoint_excitation, dict):
            excitation = deep_merge(excitation, checkpoint_excitation)

    checkpoint_path = checkpoint_dir / "checkpoint.npz"
    if checkpoint_path.exists():
        _check_excitation_artifact_frequencies(
            checkpoint_path,
            float(excitation["base_frequency_hz"]),
            frequency_key="exc_freqs",
        )

    return {
        "base_frequency_hz": float(excitation["base_frequency_hz"]),
        "trajectory_duration_periods": float(
            excitation.get("trajectory_duration_periods", 1)
        ),
    }


def _check_excitation_artifact_frequencies(path: Path,
                                           base_frequency_hz: float,
                                           frequency_key: str = "freqs") -> None:
    """Reject metadata that cannot replay the excitation artifact."""
    if not path.exists():
        return

    with np.load(str(path), allow_pickle=True) as artifact:
        if frequency_key not in artifact.files:
            return
        freqs = artifact[frequency_key]

    expected_freqs = build_frequencies(float(base_frequency_hz), freqs.size)
    if not np.allclose(freqs, expected_freqs, atol=1e-12):
        raise ValueError(
            "Validation excitation metadata does not match the excitation artifact. "
            f"Using base_frequency_hz={base_frequency_hz} expects "
            f"{expected_freqs.tolist()}, but {path} contains {freqs.tolist()}."
        )


def _is_module_available(module_name: str) -> bool:
    """Return True when a module can be imported in the current environment."""
    return importlib.util.find_spec(module_name) is not None


def _resolve_measurement_path(source: str, *, label: str) -> str:
    """Resolve a *.source path to an existing .npz measurements file.

    Accepts either a direct path to a ``.npz`` file or a directory containing
    ``measurements.npz``. Raises with a clear, layer-prefixed error so the
    caller can tell which selector failed.
    """
    path = Path(source)
    if path.is_dir():
        candidate = path / "measurements.npz"
        if not candidate.exists():
            raise FileNotFoundError(
                f"{label} is a directory but does not contain measurements.npz: "
                f"{path}"
            )
        return str(candidate.resolve())
    if path.suffix.lower() == ".npz" and path.exists():
        return str(path.resolve())
    raise FileNotFoundError(
        f"{label}={source!r} does not resolve to an existing .npz file or "
        "directory containing measurements.npz."
    )


def _existing_identification_results(cfg: dict) -> bool:
    """Return True when a previous pipeline run wrote identification_results.npz."""
    return (
        Path(cfg["output_dir"])
        / _PIPELINE_SUBDIR
        / "identification_results.npz"
    ).exists()
