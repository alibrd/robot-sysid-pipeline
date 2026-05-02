"""Top-level workflow orchestration for pipeline and optional validation stages."""
import importlib.util
import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

from .config_loader import load_config
from .config_utils import deep_merge, resolve_path_value
from .pipeline import SystemIdentificationPipeline
from .pybullet_validation import (
    PyBulletValidationRunner,
    load_default_pybullet_validation_config,
    load_pybullet_validation_config,
    normalize_pybullet_validation_config_aliases,
)
from .pybullet_validation_benchmark import export_validation_benchmark
from .pybullet_validation_report import export_validation_report
from .trajectory import build_frequencies


def load_default_workflow_config() -> dict:
    """Load the repository default workflow config."""
    default_path = Path(__file__).resolve().parent.parent / "config" / "default_workflow_config.json"
    with open(default_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_workflow_config(config_path: str) -> dict:
    """Load a workflow config and resolve relative paths against its location."""
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8-sig") as f:
        user_cfg = json.load(f)

    cfg = deep_merge(load_default_workflow_config(), user_cfg)
    cfg["_config_path"] = str(config_file)
    cfg["_config_dir"] = str(config_file.parent)
    return _resolve_workflow_paths(cfg, config_file.parent)


class WorkflowRunner:
    """Coordinate pipeline, validation, report, and benchmark stages."""

    def __init__(self, workflow_config_path: str):
        self.workflow_config_path = str(Path(workflow_config_path).resolve())
        self.cfg = load_workflow_config(self.workflow_config_path)
        self.context = None

    def prepare(self) -> dict:
        """Resolve stage configs, outputs, and preflight requirements."""
        cfg = deepcopy(self.cfg)
        workflow_dir = Path(cfg["_config_dir"])

        run_pipeline = bool(cfg.get("run_pipeline", False))
        run_validation = bool(cfg.get("run_validation", False))
        run_report = bool(cfg.get("run_report", False))
        run_benchmark = bool(cfg.get("run_benchmark", False))
        allow_missing_optional = bool(cfg.get("allow_missing_optional_dependencies", False))

        pipeline_ref = cfg.get("pipeline", {})
        validation_ref = cfg.get("validation", {})
        report_ref = cfg.get("report", {})
        benchmark_ref = cfg.get("benchmark", {})

        auto_from_pipeline = validation_ref.get("auto_from_pipeline")
        if auto_from_pipeline is None:
            auto_from_pipeline = run_pipeline and run_validation
        auto_from_pipeline = bool(auto_from_pipeline)
        validation_ref["auto_from_pipeline"] = auto_from_pipeline

        output_root = cfg.get("output_root")
        output_root_path = None
        if output_root not in (None, ""):
            output_root_path = Path(output_root)
            output_root_path.mkdir(parents=True, exist_ok=True)

        pipeline_cfg = None
        pipeline_output_dir = None
        pipeline_config_path = pipeline_ref.get("config_path")
        if run_pipeline or auto_from_pipeline:
            if not pipeline_config_path:
                raise ValueError(
                    "workflow.pipeline.config_path is required when run_pipeline=true "
                    "or validation.auto_from_pipeline=true."
                )
            pipeline_cfg = load_config(pipeline_config_path)
            if pipeline_ref.get("excitation_only"):
                pipeline_cfg["excitation_only"] = True
            if pipeline_ref.get("checkpoint_dir"):
                pipeline_cfg["checkpoint_dir"] = pipeline_ref["checkpoint_dir"]
            if output_root_path is not None:
                pipeline_cfg["output_dir"] = str(
                    output_root_path / "pipeline" / Path(pipeline_config_path).stem
                )
            pipeline_output_dir = Path(pipeline_cfg["output_dir"])

        validation_cfg = None
        validation_output_root = None
        validation_config_path = validation_ref.get("config_path")
        if run_validation:
            validation_base = load_default_pybullet_validation_config()
            if validation_config_path:
                validation_base = load_pybullet_validation_config(validation_config_path)

            inline_validation = _extract_inline_validation_fields(validation_ref)
            inline_validation = normalize_pybullet_validation_config_aliases(
                inline_validation,
                config_path="workflow.validation",
            )
            inline_validation = _resolve_validation_inline_paths(inline_validation, workflow_dir)
            validation_cfg = deep_merge(validation_base, inline_validation)

            if auto_from_pipeline:
                if pipeline_cfg is None or pipeline_output_dir is None:
                    raise ValueError(
                        "validation.auto_from_pipeline=true requires a resolved pipeline config."
                    )
                validation_cfg["urdf_path"] = pipeline_cfg["urdf_path"]
                validation_cfg["excitation_file"] = str(
                    pipeline_output_dir / "excitation_trajectory.npz"
                )
                excitation_meta = _pipeline_validation_excitation_metadata(
                    pipeline_cfg
                )
                validation_cfg["base_frequency_hz"] = (
                    excitation_meta["base_frequency_hz"]
                )
                validation_cfg["trajectory_duration_periods"] = (
                    excitation_meta["trajectory_duration_periods"]
                )
                friction_model = pipeline_cfg.get("friction", {}).get("model", "none")
                if friction_model != "none":
                    warnings.warn(
                        "Workflow auto_from_pipeline is deriving a PyBullet validation run "
                        f"from a pipeline config with friction.model='{friction_model}', "
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

            if output_root_path is not None:
                validation_cfg["output_dir"] = str(output_root_path / "validation")

            if not validation_cfg.get("urdf_path"):
                raise ValueError("Validation config must define 'urdf_path'.")
            if not validation_cfg.get("excitation_file"):
                raise ValueError("Validation config must define 'excitation_file'.")

            validation_output_root = Path(validation_cfg["output_dir"])

        report_validation_dir = report_ref.get("validation_dir")
        if run_report and not run_validation:
            if not report_validation_dir:
                raise ValueError(
                    "workflow.report.validation_dir is required when run_report=true "
                    "and run_validation=false."
                )
            report_validation_dir = str(Path(report_validation_dir))

        benchmark_validation_root = benchmark_ref.get("validation_root")
        if run_benchmark:
            if benchmark_validation_root:
                benchmark_validation_root = str(Path(benchmark_validation_root))
            elif output_root_path is not None:
                benchmark_validation_root = str(output_root_path / "validation")
            elif run_validation and validation_output_root is not None:
                benchmark_validation_root = str(validation_output_root)
            else:
                raise ValueError(
                    "workflow.benchmark.validation_root is required when run_benchmark=true "
                    "and it cannot be inferred from output_root or the validation stage."
                )

        _check_validation_consistency(
            run_pipeline=run_pipeline,
            run_validation=run_validation,
            auto_from_pipeline=auto_from_pipeline,
            pipeline_cfg=pipeline_cfg,
            pipeline_output_dir=pipeline_output_dir,
            validation_cfg=validation_cfg,
            use_external_artifacts=bool(validation_ref.get("use_external_artifacts", False)),
        )

        if run_validation and not run_pipeline:
            excitation_path = Path(validation_cfg["excitation_file"])
            if not excitation_path.exists():
                raise FileNotFoundError(
                    "Validation requires an existing excitation artifact when "
                    f"run_pipeline=false. Missing file: {excitation_path}"
                )

        if run_validation and not _is_module_available("pybullet"):
            if allow_missing_optional:
                run_validation = False
            else:
                raise RuntimeError(
                    "PyBullet is required for the validation stage but is not installed."
                )

        if run_report and not run_validation and not report_validation_dir:
            if allow_missing_optional:
                run_report = False
            else:
                raise RuntimeError(
                    "The report stage could not infer report.validation_dir because "
                    "the validation stage is disabled."
                )

        if run_report and not _is_module_available("matplotlib"):
            if allow_missing_optional:
                run_report = False
            else:
                raise RuntimeError(
                    "matplotlib is required for the report stage but is not installed."
                )

        context = {
            "workflow_config_path": self.workflow_config_path,
            "run_pipeline": run_pipeline,
            "run_validation": run_validation,
            "run_report": run_report,
            "run_benchmark": run_benchmark,
            "auto_from_pipeline": auto_from_pipeline,
            "pipeline_cfg": pipeline_cfg,
            "validation_cfg": validation_cfg,
            "report_validation_dir": report_validation_dir,
            "benchmark_validation_root": benchmark_validation_root,
        }
        self.context = context
        return context

    def run(self) -> dict:
        """Execute the configured workflow stages in order."""
        ctx = self.prepare() if self.context is None else self.context
        results = {}

        pipeline_runner = None
        if ctx["run_pipeline"]:
            pipeline_cfg = ctx["pipeline_cfg"]
            # Inject workflow-level pipeline partitioning overrides
            pipeline_section = self.cfg.get("pipeline", {})
            if pipeline_section.get("excitation_only"):
                pipeline_cfg["excitation_only"] = True
            cp_dir = pipeline_section.get("checkpoint_dir")
            if cp_dir:
                pipeline_cfg["checkpoint_dir"] = cp_dir
            pipeline_runner = SystemIdentificationPipeline(pipeline_cfg)
            pipeline_runner.run()
            results["pipeline_output_dir"] = str(pipeline_runner.output_dir)
            if ctx["run_validation"] and ctx["auto_from_pipeline"]:
                _check_excitation_artifact_frequencies(
                    Path(ctx["validation_cfg"]["excitation_file"]),
                    ctx["validation_cfg"]["base_frequency_hz"],
                )

        validation_runner = None
        if ctx["run_validation"]:
            validation_runner = PyBulletValidationRunner(ctx["validation_cfg"])
            results["validation_summary"] = validation_runner.run()
            results["validation_output_dir"] = str(validation_runner.output_dir)

        if ctx["run_report"]:
            validation_dir = ctx["report_validation_dir"]
            if not validation_dir:
                if validation_runner is None or validation_runner.output_dir is None:
                    raise RuntimeError(
                        "Could not infer report.validation_dir because the validation stage "
                        "did not produce an output directory."
                    )
                validation_dir = str(validation_runner.output_dir)
            results["report"] = export_validation_report(validation_dir)

        if ctx["run_benchmark"]:
            results["benchmark"] = export_validation_benchmark(
                ctx["benchmark_validation_root"],
                ctx["benchmark_validation_root"],
            )

        return results


def _extract_inline_validation_fields(validation_cfg: dict) -> dict:
    """Extract validation config keys that belong to the validation stage itself."""
    reserved = {"config_path", "auto_from_pipeline", "use_external_artifacts"}
    return {
        key: deepcopy(value)
        for key, value in validation_cfg.items()
        if key not in reserved
    }


def _pipeline_validation_excitation_metadata(pipeline_cfg: dict) -> dict:
    """Return excitation metadata matching the generated validation artifact."""
    excitation = deepcopy(pipeline_cfg["excitation"])
    checkpoint_dir = pipeline_cfg.get("checkpoint_dir")
    if checkpoint_dir:
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
                excitation["base_frequency_hz"],
                frequency_key="exc_freqs",
            )

    return {
        "base_frequency_hz": excitation["base_frequency_hz"],
        "trajectory_duration_periods": excitation.get(
            "trajectory_duration_periods", 1
        ),
    }


def _check_excitation_artifact_frequencies(path: Path,
                                           base_frequency_hz: float,
                                           frequency_key: str = "freqs") -> None:
    """Reject validation metadata that cannot replay the excitation artifact."""
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


def _check_validation_consistency(*,
                                  run_pipeline: bool,
                                  run_validation: bool,
                                  auto_from_pipeline: bool,
                                  pipeline_cfg: dict | None,
                                  pipeline_output_dir: Path | None,
                                  validation_cfg: dict | None,
                                  use_external_artifacts: bool) -> None:
    """Reject mismatched cross-stage inputs before any stage runs."""
    if not run_validation or validation_cfg is None or pipeline_cfg is None:
        return
    if auto_from_pipeline:
        return

    pipeline_urdf = Path(pipeline_cfg["urdf_path"]).resolve()
    validation_urdf = Path(validation_cfg["urdf_path"]).resolve()
    if pipeline_urdf != validation_urdf:
        raise ValueError(
            "Pipeline and validation URDF paths differ. "
            f"Pipeline uses {pipeline_urdf}, validation uses {validation_urdf}."
        )

    if run_pipeline and not use_external_artifacts:
        expected_excitation = (pipeline_output_dir / "excitation_trajectory.npz").resolve()
        validation_excitation = Path(validation_cfg["excitation_file"]).resolve()
        if validation_excitation != expected_excitation:
            raise ValueError(
                "Validation excitation_file does not match the pipeline output artifact. "
                f"Expected {expected_excitation}, got {validation_excitation}. "
                "Set validation.use_external_artifacts=true if this is intentional."
            )


def _resolve_workflow_paths(cfg: dict, base_dir: Path) -> dict:
    """Resolve workflow-level path references relative to the workflow config file."""
    resolved = deepcopy(cfg)
    resolved["output_root"] = resolve_path_value(resolved.get("output_root"), base_dir)

    for section_name, keys in {
        "pipeline": ("config_path", "checkpoint_dir"),
        "validation": ("config_path", "urdf_path", "excitation_file", "output_dir"),
        "report": ("validation_dir",),
        "benchmark": ("validation_root",),
    }.items():
        section = deepcopy(resolved.get(section_name, {}))
        for key in keys:
            if key in section:
                section[key] = resolve_path_value(section.get(key), base_dir)
        resolved[section_name] = section

    return resolved


def _resolve_validation_inline_paths(cfg: dict, base_dir: Path) -> dict:
    """Resolve path-valued inline validation fields relative to the workflow config."""
    resolved = deepcopy(cfg)
    for key in ("urdf_path", "excitation_file", "output_dir"):
        if key in resolved:
            resolved[key] = resolve_path_value(resolved.get(key), base_dir)
    return resolved


def _is_module_available(module_name: str) -> bool:
    """Return True when a module can be imported in the current environment."""
    return importlib.util.find_spec(module_name) is not None
