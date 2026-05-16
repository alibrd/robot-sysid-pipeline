"""Validate an identified rigid-body + friction model against real measurements.

Robot-agnostic: works with any URDF and any pipeline-compatible measurement
``.npz`` (keys: ``q``, ``dq``, ``ddq``, ``tau``; optional ``fs``,
``joint_names``). The measurement file must already be in pipeline-ready form;
filtering, finite-differencing, trimming, and downsampling are the user's
responsibility and live outside the pipeline.

This module is invoked by :class:`src.runner.UnifiedRunner` whenever
``validation.source`` is a filesystem path (sentinel ``"pybullet"`` dispatches
the existing :class:`PyBulletValidationRunner` instead).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .dynamics_newton_euler import newton_euler_regressor
from .friction import build_friction_regressor, friction_param_count
from .kinematics import RobotKinematics
from .pipeline_logger import setup_logger
from .urdf_parser import parse_urdf


_SUMMARY_FILENAME = "measurement_validation_summary.json"
_REPORT_FILENAME = "measurement_validation_report.md"
_METRICS_CSV_FILENAME = "measurement_validation_metrics.csv"
_DATA_FILENAME = "measurement_validation_data.npz"


class MeasurementValidationRunner:
    """Compare an identified model's torque prediction against measured torque."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(str(self.output_dir))

    def run(self) -> dict:
        log = self.logger
        cfg = self.cfg

        log.info("=" * 60)
        log.info("MEASUREMENT VALIDATION")
        log.info("=" * 60)
        log.info("URDF:         %s", cfg["urdf_path"])
        log.info("Measurements: %s", cfg["measurements_path"])
        log.info("Pipeline dir: %s", cfg["pipeline_dir"])

        measurements = _load_measurements(cfg["measurements_path"])
        robot = parse_urdf(cfg["urdf_path"])
        n_dof = robot.nDoF
        if measurements["q"].shape[1] != n_dof:
            raise ValueError(
                "Measurement DoF count does not match URDF. "
                f"q has {measurements['q'].shape[1]} columns, "
                f"URDF reports nDoF={n_dof}."
            )
        kin = RobotKinematics(robot, log)
        joint_names = list(robot.revolute_joint_names)

        model = _load_identified_model(
            Path(cfg["pipeline_dir"]),
            expected_dof=n_dof,
            fallback_method=cfg.get("method", "newton_euler"),
            fallback_friction=cfg.get("friction_model", "none"),
        )

        if model["method"] != "newton_euler":
            raise NotImplementedError(
                "MeasurementValidationRunner currently supports only "
                f"method='newton_euler'; identified model uses {model['method']!r}."
            )

        expected_param_count = n_dof * 10 + friction_param_count(
            n_dof, model["friction_model"]
        )
        if model["params"].size != expected_param_count:
            raise ValueError(
                "Identified parameter vector length does not match the URDF + "
                "friction-model structure. "
                f"Expected {expected_param_count}, got {model['params'].size}. "
                f"friction_model={model['friction_model']!r}, nDoF={n_dof}."
            )

        log.info(
            "Computing identified-model inverse dynamics for %d samples",
            measurements["q"].shape[0],
        )
        tau_model = _compute_model_torque(
            kin=kin,
            q=measurements["q"],
            dq=measurements["dq"],
            ddq=measurements["ddq"],
            params=model["params"],
            friction_model=model["friction_model"],
            logger=log,
        )

        comparison = _compute_metrics(
            tau_reference=measurements["tau"],
            tau_candidate=tau_model,
            joint_names=joint_names,
        )

        summary = self._write_outputs(
            measurements=measurements,
            tau_model=tau_model,
            comparison=comparison,
            model=model,
            joint_names=joint_names,
        )
        log.info(
            "Measurement validation complete. Global RMSE %.6g Nm, normalised RMSE %.6g",
            comparison["global"]["rmse"],
            comparison["global"]["normalized_rmse"],
        )
        log.info("Outputs written to %s", self.output_dir)
        return summary

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_outputs(self,
                       *,
                       measurements: dict,
                       tau_model: np.ndarray,
                       comparison: dict,
                       model: dict,
                       joint_names: list[str]) -> dict:
        out_dir = self.output_dir
        np.savez(
            str(out_dir / _DATA_FILENAME),
            t=measurements["t"],
            q=measurements["q"],
            dq=measurements["dq"],
            ddq=measurements["ddq"],
            tau_measured=measurements["tau"],
            tau_model=tau_model,
            tau_error=comparison["tau_error"],
            tau_abs_error=comparison["tau_abs_error"],
            joint_names=np.asarray(joint_names, dtype=object),
        )

        summary = {
            "passed": _decide_pass(comparison),
            "model": _json_safe_model(model),
            "urdf_path": str(self.cfg["urdf_path"]),
            "measurements_path": str(self.cfg["measurements_path"]),
            "sample_count": int(measurements["q"].shape[0]),
            "sample_rate_hz": float(measurements["fs"]),
            "joint_names": joint_names,
            "global_metrics": comparison["global"],
            "per_joint_metrics": comparison["per_joint"],
            "notes": [
                "Open-loop inverse-dynamics validation. No refitting, offset, "
                "delay, or sign correction is applied to the identified parameters.",
                "Inputs (q, dq, ddq) are taken verbatim from the supplied "
                "measurement .npz; any filtering/finite-differencing is the "
                "user's responsibility and happens outside the pipeline.",
            ],
        }
        (out_dir / _SUMMARY_FILENAME).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _write_metrics_csv(out_dir / _METRICS_CSV_FILENAME, comparison["per_joint"])
        _write_markdown_report(out_dir / _REPORT_FILENAME, summary)
        return summary


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_measurements(path: str | Path) -> dict[str, Any]:
    """Load a pipeline-compatible measurements .npz."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Measurement file not found: {p}")
    with np.load(str(p), allow_pickle=True) as data:
        required = {"q", "dq", "ddq", "tau"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Measurement .npz {p} is missing required keys: "
                f"{sorted(missing)}. Required: {sorted(required)}."
            )
        q = np.asarray(data["q"], dtype=float)
        dq = np.asarray(data["dq"], dtype=float)
        ddq = np.asarray(data["ddq"], dtype=float)
        tau = np.asarray(data["tau"], dtype=float)
        fs = float(np.asarray(data["fs"]).item()) if "fs" in data.files else 0.0

    if not (q.shape == dq.shape == ddq.shape == tau.shape):
        raise ValueError(
            "Measurement arrays must all share the same shape. Got "
            f"q={q.shape}, dq={dq.shape}, ddq={ddq.shape}, tau={tau.shape}."
        )
    n_samples = q.shape[0]
    if fs > 0.0:
        t = np.arange(n_samples, dtype=float) / fs
    else:
        t = np.arange(n_samples, dtype=float)
    return {"q": q, "dq": dq, "ddq": ddq, "tau": tau, "fs": fs, "t": t}


def _load_identified_model(pipeline_dir: Path,
                           *,
                           expected_dof: int,
                           fallback_method: str,
                           fallback_friction: str) -> dict[str, Any]:
    """Load identified parameters from a previous pipeline run's output."""
    result_path = pipeline_dir / "identification_results.npz"
    if not result_path.exists():
        raise FileNotFoundError(
            "Identified-model file not found. Expected "
            f"{result_path}. Run identification (stages.identification=true) "
            "before running measurement validation."
        )

    with np.load(str(result_path), allow_pickle=True) as data:
        if "pi_corrected" in data.files:
            param_source = "pi_corrected"
            params = np.asarray(data["pi_corrected"], dtype=float).reshape(-1)
        elif "pi_identified" in data.files:
            param_source = "pi_identified"
            params = np.asarray(data["pi_identified"], dtype=float).reshape(-1)
        else:
            raise ValueError(
                "identification_results.npz must contain pi_corrected or pi_identified."
            )

        n_dof = (
            int(np.asarray(data["nDoF"]).item())
            if "nDoF" in data.files
            else expected_dof
        )
        if n_dof != expected_dof:
            raise ValueError(
                f"Identified model nDoF={n_dof} does not match URDF nDoF={expected_dof}."
            )
        method = (
            _np_scalar_to_str(data["method"])
            if "method" in data.files
            else fallback_method
        )
        friction_model = (
            _np_scalar_to_str(data["friction_model"])
            if "friction_model" in data.files
            else fallback_friction
        )

    return {
        "path": str(result_path),
        "pipeline_dir": str(pipeline_dir),
        "params": params,
        "param_source": param_source,
        "method": method,
        "friction_model": friction_model,
        "nDoF": n_dof,
    }


def _np_scalar_to_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _compute_model_torque(*,
                          kin: RobotKinematics,
                          q: np.ndarray,
                          dq: np.ndarray,
                          ddq: np.ndarray,
                          params: np.ndarray,
                          friction_model: str,
                          logger: logging.Logger) -> np.ndarray:
    tau_model = np.zeros((q.shape[0], kin.nDoF), dtype=float)
    progress_step = max(1, q.shape[0] // 10)
    for idx in range(q.shape[0]):
        rigid = newton_euler_regressor(kin, q[idx], dq[idx], ddq[idx])
        if friction_model != "none":
            friction = build_friction_regressor(dq[idx], friction_model)
            regressor = np.hstack((rigid, friction))
        else:
            regressor = rigid
        tau_model[idx] = regressor @ params
        if (idx + 1) % progress_step == 0 or idx + 1 == q.shape[0]:
            logger.info("  torque samples: %d / %d", idx + 1, q.shape[0])
    return tau_model


def _compute_metrics(*,
                     tau_reference: np.ndarray,
                     tau_candidate: np.ndarray,
                     joint_names: list[str]) -> dict[str, Any]:
    error = tau_candidate - tau_reference
    abs_error = np.abs(error)
    eps = 1e-12

    per_joint = []
    for j, joint_name in enumerate(joint_names):
        ref = tau_reference[:, j]
        pred = tau_candidate[:, j]
        err = error[:, j]
        rmse = float(np.sqrt(np.mean(err**2)))
        ref_rms = float(np.sqrt(np.mean(ref**2)))
        ref_peak = float(np.max(np.abs(ref)))
        per_joint.append({
            "joint_name": joint_name,
            "rmse": rmse,
            "reference_rms": ref_rms,
            "reference_peak_abs": ref_peak,
            "normalized_rmse": float(rmse / max(ref_rms, eps)),
            "mae": float(np.mean(np.abs(err))),
            "max_abs_error": float(np.max(np.abs(err))),
            "p95_abs_error": float(np.percentile(np.abs(err), 95)),
            "mean_bias": float(np.mean(err)),
            "std_error": float(np.std(err, ddof=0)),
            "correlation": _safe_correlation(ref, pred),
            "r2_score": _r2_score(ref, pred),
            "rmse_pct_of_peak_reference": float(
                100.0 * rmse / max(ref_peak, eps)
            ),
        })

    flat_ref = tau_reference.reshape(-1)
    flat_pred = tau_candidate.reshape(-1)
    global_rmse = float(np.sqrt(np.mean(error**2)))
    global_ref_rms = float(np.sqrt(np.mean(tau_reference**2)))
    global_ref_peak = float(np.max(np.abs(tau_reference)))
    global_metrics = {
        "rmse": global_rmse,
        "reference_rms": global_ref_rms,
        "reference_peak_abs": global_ref_peak,
        "normalized_rmse": float(global_rmse / max(global_ref_rms, eps)),
        "mae": float(np.mean(abs_error)),
        "max_abs_error": float(np.max(abs_error)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "mean_bias": float(np.mean(error)),
        "std_error": float(np.std(error, ddof=0)),
        "correlation": _safe_correlation(flat_ref, flat_pred),
        "r2_score": _r2_score(flat_ref, flat_pred),
        "rmse_pct_of_peak_reference": float(
            100.0 * global_rmse / max(global_ref_peak, eps)
        ),
    }
    return {
        "tau_error": error,
        "tau_abs_error": abs_error,
        "per_joint": per_joint,
        "global": global_metrics,
    }


def _decide_pass(comparison: dict) -> bool:
    """Lenient pass criterion: identified model must outperform a zero predictor."""
    g = comparison["global"]
    return g["normalized_rmse"] < 1.0


def _r2_score(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    ss_res = float(np.sum((reference - candidate) ** 2))
    ss_tot = float(np.sum((reference - np.mean(reference)) ** 2))
    if ss_tot <= 1e-24:
        return None
    return float(1.0 - ss_res / ss_tot)


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _json_safe_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": model["path"],
        "pipeline_dir": model["pipeline_dir"],
        "parameter_source": model["param_source"],
        "method": model["method"],
        "friction_model": model["friction_model"],
        "nDoF": int(model["nDoF"]),
        "parameter_count": int(model["params"].size),
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "joint_name", "rmse", "reference_rms", "reference_peak_abs",
        "normalized_rmse", "mae", "max_abs_error", "p95_abs_error",
        "mean_bias", "std_error", "correlation", "r2_score",
        "rmse_pct_of_peak_reference",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    g = summary["global_metrics"]
    lines = [
        "# Measurement Validation Report",
        "",
        "## Summary",
        "",
        f"- URDF: `{summary['urdf_path']}`",
        f"- Measurements: `{summary['measurements_path']}`",
        f"- Identified model: `{summary['model']['path']}`",
        f"- Parameter source: `{summary['model']['parameter_source']}`",
        f"- Method: `{summary['model']['method']}`",
        f"- Friction model: `{summary['model']['friction_model']}`",
        f"- Samples: `{summary['sample_count']}` "
        f"(fs = `{summary['sample_rate_hz']}` Hz)",
        "",
        "### Global Torque Errors",
        "",
        f"- RMSE: `{g['rmse']:.6g}` Nm",
        f"- Normalised RMSE: `{g['normalized_rmse']:.6g}`",
        f"- MAE: `{g['mae']:.6g}` Nm",
        f"- p95 |err|: `{g['p95_abs_error']:.6g}` Nm",
        f"- Max |err|: `{g['max_abs_error']:.6g}` Nm",
        f"- Mean bias: `{g['mean_bias']:.6g}` Nm",
        f"- Std error: `{g['std_error']:.6g}` Nm",
        f"- Reference peak |τ|: `{g['reference_peak_abs']:.6g}` Nm",
        f"- RMSE / peak ref: `{g['rmse_pct_of_peak_reference']:.4g}` %",
        f"- Correlation: `{_fmt(g['correlation'])}`",
        f"- R²: `{_fmt(g['r2_score'])}`",
        "",
        "## Per-Joint Errors",
        "",
        "| Joint | RMSE | nRMSE | MAE | p95|err| | Max|err| | Bias | Std | Corr | R² | RMSE/peak% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_joint_metrics"]:
        lines.append(
            f"| {row['joint_name']} | {row['rmse']:.6g} | "
            f"{row['normalized_rmse']:.6g} | {row['mae']:.6g} | "
            f"{row['p95_abs_error']:.6g} | {row['max_abs_error']:.6g} | "
            f"{row['mean_bias']:.6g} | {row['std_error']:.6g} | "
            f"{_fmt(row['correlation'])} | {_fmt(row['r2_score'])} | "
            f"{row['rmse_pct_of_peak_reference']:.4g} |"
        )
    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"
