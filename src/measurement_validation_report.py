"""Generate PNG plots for a measurement-validation run.

The markdown summary and per-joint CSV are written by
:class:`MeasurementValidationRunner` itself; this module only adds the plot
pack that the ``stages.report`` flag is expected to produce, mirroring the
PyBullet exporter's behaviour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pybullet_validation_report import _write_torque_plots


_SUMMARY_FILENAME = "measurement_validation_summary.json"
_DATA_FILENAME = "measurement_validation_data.npz"
_REPORT_FILENAME = "measurement_validation_report.md"
_METRICS_CSV_FILENAME = "measurement_validation_metrics.csv"


def export_measurement_validation_report(validation_dir: str) -> dict:
    """Generate plots for a measurement-validation run.

    The markdown report and per-joint CSV are already written during
    validation; this function adds torque-overlay and absolute-error PNGs
    using the shared plotting helper.
    """
    base = Path(validation_dir)
    summary_path = base / _SUMMARY_FILENAME
    data_path = base / _DATA_FILENAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Validation summary not found: {summary_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {data_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    data = np.load(str(data_path))

    plot_paths = _write_torque_plots(
        out_dir=base,
        joint_names=summary["joint_names"],
        t=data["t"],
        tau_a=data["tau_measured"],
        tau_b=data["tau_model"],
        tau_abs_error=data["tau_abs_error"],
        label_a="Measured",
        label_b="Identified model",
    )

    return {
        "report_markdown": str(base / _REPORT_FILENAME),
        "metrics_csv": str(base / _METRICS_CSV_FILENAME),
        "plot_paths": plot_paths,
    }
