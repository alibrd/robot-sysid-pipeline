"""Aggregate multiple PyBullet validation runs into benchmark tables."""
import csv
from pathlib import Path

from .pybullet_validation_report import load_validation_summary


def discover_validation_runs(root_dir: str) -> list[Path]:
    """Recursively find validation run directories under *root_dir*."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Validation root not found: {root}")

    if (root / "pybullet_validation_summary.json").exists():
        return [root]

    run_dirs = sorted(
        {
            summary_path.parent
            for summary_path in root.rglob("pybullet_validation_summary.json")
        },
        key=lambda path: str(path).lower(),
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No validation runs found under: {root}. Expected at least one "
            "'pybullet_validation_summary.json'."
        )
    return run_dirs


def export_validation_benchmark(root_dir: str, output_dir: str | None = None) -> dict:
    """Write a benchmark CSV and Markdown summary for all runs under *root_dir*."""
    root = Path(root_dir)
    run_dirs = discover_validation_runs(str(root))
    out_dir = Path(output_dir) if output_dir is not None else root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_benchmark_rows(run_dirs, root)
    csv_path = out_dir / "pybullet_validation_benchmark.csv"
    md_path = out_dir / "pybullet_validation_benchmark.md"
    _write_benchmark_csv(csv_path, rows)
    _write_benchmark_markdown(md_path, rows)
    return {
        "benchmark_csv": str(csv_path),
        "benchmark_markdown": str(md_path),
        "run_count": len(rows),
    }


def _build_benchmark_rows(run_dirs: list[Path], root: Path):
    """Build sorted benchmark rows from run summaries."""
    rows = []
    for run_dir in run_dirs:
        summary = load_validation_summary(str(run_dir))
        rel_dir = str(run_dir.relative_to(root)) if run_dir != root else "."
        rows.append({
            "robot_name": summary["robot_name"],
            "run_dir": rel_dir,
            "nDoF": int(summary["nDoF"]),
            "sample_count": int(summary["sample_count"]),
            "sample_rate_hz": float(summary["sample_rate_hz"]),
            "tolerance_abs": float(summary["tolerance_abs"]),
            "tolerance_normalized_rms": float(summary["tolerance_normalized_rms"]),
            "global_max_abs_error": float(summary["global_max_abs_error"]),
            "global_rms_error": float(summary["global_rms_error"]),
            "global_normalized_rms_error": float(summary["global_normalized_rms_error"]),
            "worst_joint_max_abs_error": float(max(summary["max_abs_error_per_joint"])),
            "worst_joint_normalized_rms_error": float(
                max(summary["normalized_rms_error_per_joint"])
            ),
            "passed": bool(summary["passed"]),
        })

    rows.sort(key=lambda row: (not row["passed"], row["robot_name"].lower(),
                               row["run_dir"].lower()))
    return rows


def _write_benchmark_csv(csv_path: Path, rows):
    """Write the flat benchmark CSV."""
    fieldnames = [
        "robot_name",
        "run_dir",
        "nDoF",
        "sample_count",
        "sample_rate_hz",
        "tolerance_abs",
        "tolerance_normalized_rms",
        "global_max_abs_error",
        "global_rms_error",
        "global_normalized_rms_error",
        "worst_joint_max_abs_error",
        "worst_joint_normalized_rms_error",
        "passed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_markdown(md_path: Path, rows):
    """Write the benchmark markdown summary."""
    pass_count = sum(1 for row in rows if row["passed"])
    fail_count = len(rows) - pass_count
    worst_max_abs = max(row["global_max_abs_error"] for row in rows)
    worst_norm_rms = max(row["global_normalized_rms_error"] for row in rows)

    lines = [
        "# PyBullet Validation Benchmark",
        "",
        f"- Runs aggregated: `{len(rows)}`",
        f"- Pass count: `{pass_count}`",
        f"- Fail count: `{fail_count}`",
        f"- Worst global max absolute error: `{worst_max_abs:.12g}`",
        f"- Worst global normalized RMS error: `{worst_norm_rms:.12g}`",
        "",
        "| Robot | Run dir | DoF | Samples | Max abs error | RMS error | Normalized RMS error | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['robot_name']} | {row['run_dir']} | {row['nDoF']} | "
            f"{row['sample_count']} | {row['global_max_abs_error']:.12g} | "
            f"{row['global_rms_error']:.12g} | "
            f"{row['global_normalized_rms_error']:.12g} | "
            f"{'yes' if row['passed'] else 'no'} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
