"""Generate report-ready tables and plots from PyBullet validation outputs."""
import csv
import json
import importlib
from pathlib import Path

import numpy as np


def load_validation_summary(validation_dir: str) -> dict:
    """Load only the validation summary JSON for a run directory."""
    base = Path(validation_dir)
    summary_path = base / "pybullet_validation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Validation summary not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if "tolerance_normalized_rms" not in summary and "tolerance_rel" in summary:
        summary["tolerance_normalized_rms"] = summary["tolerance_rel"]
    return summary


def load_validation_artifacts(validation_dir: str):
    """Load the summary JSON and data NPZ for a validation run."""
    base = Path(validation_dir)
    data_path = base / "pybullet_validation_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {data_path}")

    summary = load_validation_summary(validation_dir)
    data = np.load(data_path)
    return summary, data


def export_validation_report(validation_dir: str) -> dict:
    """Write a markdown summary, a CSV table, and PNG plots for a validation run.

    Dispatches to the PyBullet or measurement-path exporter depending on which
    summary file the validation backend produced in ``validation_dir``.
    """
    base = Path(validation_dir)
    if (base / "pybullet_validation_summary.json").exists():
        return _export_pybullet_report(base)
    if (base / "measurement_validation_summary.json").exists():
        from .measurement_validation_report import export_measurement_validation_report
        return export_measurement_validation_report(validation_dir)
    raise FileNotFoundError(
        "No validation summary found in "
        f"{validation_dir}. Expected pybullet_validation_summary.json or "
        "measurement_validation_summary.json."
    )


def _export_pybullet_report(out_dir: Path) -> dict:
    summary, data = load_validation_artifacts(str(out_dir))

    csv_path = out_dir / "pybullet_validation_metrics.csv"
    md_path = out_dir / "pybullet_validation_report.md"

    rows = _build_metric_rows(summary)
    _write_metrics_csv(csv_path, rows)
    _write_markdown_report(md_path, summary, rows)
    plot_paths = _write_torque_plots(
        out_dir=out_dir,
        joint_names=summary["joint_names"],
        t=data["t"],
        tau_a=data["tau_pipeline"],
        tau_b=data["tau_pybullet"],
        tau_abs_error=data["tau_abs_error"],
        label_a="Pipeline",
        label_b="PyBullet",
    )

    return {
        "report_markdown": str(md_path),
        "metrics_csv": str(csv_path),
        "plot_paths": plot_paths,
    }


def _build_metric_rows(summary: dict):
    """Build a flat per-joint metrics table."""
    rows = []
    for joint_name, max_abs, rms, norm_rms, passed in zip(
        summary["joint_names"],
        summary["max_abs_error_per_joint"],
        summary["rms_error_per_joint"],
        summary["normalized_rms_error_per_joint"],
        summary["per_joint_pass"],
    ):
        rows.append({
            "joint_name": joint_name,
            "max_abs_error": float(max_abs),
            "rms_error": float(rms),
            "normalized_rms_error": float(norm_rms),
            "passed": bool(passed),
        })
    return rows


def _write_metrics_csv(csv_path: Path, rows):
    """Write the flat per-joint metrics CSV."""
    fieldnames = [
        "joint_name",
        "max_abs_error",
        "rms_error",
        "normalized_rms_error",
        "passed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(md_path: Path, summary: dict, rows):
    """Write a compact report-ready markdown summary."""
    lines = [
        f"# PyBullet Validation Report: {summary['robot_name']}",
        "",
        f"- Overall result: `{'PASS' if summary['passed'] else 'FAIL'}`",
        f"- DoF: `{summary['nDoF']}`",
        f"- Samples: `{summary['sample_count']}` at `{summary['sample_rate_hz']:.3f} Hz`",
        f"- Absolute tolerance: `{summary['tolerance_abs']}`",
        f"- Normalized RMS tolerance: `{summary['tolerance_normalized_rms']}`",
        f"- Global max absolute error: `{summary['global_max_abs_error']:.12g}`",
        f"- Global RMS error: `{summary['global_rms_error']:.12g}`",
        f"- Global normalized RMS error: `{summary['global_normalized_rms_error']:.12g}`",
        "",
        "## Per-joint metrics",
        "",
        "| Joint | Max abs error | RMS error | Normalized RMS error | Pass |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['joint_name']} | {row['max_abs_error']:.12g} | "
            f"{row['rms_error']:.12g} | {row['normalized_rms_error']:.12g} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
    ])
    for note in summary.get("notes", []):
        lines.append(f"- {note}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_torque_plots(
    *,
    out_dir: Path,
    joint_names,
    t,
    tau_a,
    tau_b,
    tau_abs_error,
    label_a: str,
    label_b: str,
) -> list[str]:
    """Generate torque overlay and absolute error plots for each joint."""
    plt = _import_matplotlib_pyplot()
    plot_paths = []
    err_ylabel = f"|tau_{label_b.lower().replace(' ', '_')} - tau_{label_a.lower().replace(' ', '_')}|"

    combined_overlay_path = out_dir / "torque_overlay_all_joints.png"
    fig, axes = plt.subplots(
        nrows=len(joint_names),
        ncols=1,
        figsize=(10, max(4.0, 3.6 * len(joint_names))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for joint_idx, joint_name in enumerate(joint_names):
        ax = axes[joint_idx]
        ax.plot(t, tau_a[:, joint_idx], label=label_a, linewidth=1.6)
        ax.plot(t, tau_b[:, joint_idx], label=label_b, linewidth=1.2,
                linestyle="--")
        ax.set_title(f"Torque Overlay: {joint_name}")
        ax.set_ylabel("Torque")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(combined_overlay_path, dpi=160)
    plt.close(fig)
    plot_paths.append(str(combined_overlay_path))

    for joint_idx, joint_name in enumerate(joint_names):
        overlay_path = out_dir / f"torque_overlay_{joint_name}.png"
        error_path = out_dir / f"torque_abs_error_{joint_name}.png"

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(t, tau_a[:, joint_idx], label=label_a, linewidth=1.6)
        ax.plot(t, tau_b[:, joint_idx], label=label_b, linewidth=1.2,
                linestyle="--")
        ax.set_title(f"Torque Overlay: {joint_name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Torque")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4.0))
        ax.plot(t, tau_abs_error[:, joint_idx], color="#c23b22", linewidth=1.4)
        ax.set_title(f"Absolute Torque Error: {joint_name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(err_ylabel)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(error_path, dpi=160)
        plt.close(fig)

        plot_paths.extend([str(overlay_path), str(error_path)])

    return plot_paths


def _import_matplotlib_pyplot():
    """Import matplotlib lazily so plotting stays optional."""
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for report export but is not installed. "
            "Install it in the active environment and retry."
        ) from exc
