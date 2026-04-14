#!/usr/bin/env python3
"""Entry point for exporting a multi-run PyBullet validation benchmark."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pybullet_validation_benchmark import export_validation_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple PyBullet validation runs into a benchmark table"
    )
    parser.add_argument(
        "validation_root",
        type=str,
        help="Root directory containing one or more validation run folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for the benchmark CSV/Markdown outputs",
    )
    args = parser.parse_args()
    export_validation_benchmark(args.validation_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
