#!/usr/bin/env python3
"""Entry point for exporting PyBullet validation report artifacts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pybullet_validation_report import export_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export plots and tables from a PyBullet validation run"
    )
    parser.add_argument(
        "validation_dir",
        type=str,
        help="Directory containing pybullet_validation_summary.json and pybullet_validation_data.npz",
    )
    args = parser.parse_args()
    export_validation_report(args.validation_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
