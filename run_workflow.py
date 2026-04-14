#!/usr/bin/env python3
"""Entry point for the optional workflow orchestration layer."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.workflow import WorkflowRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pipeline, optional PyBullet validation, reports, and benchmark"
    )
    parser.add_argument("config", type=str, help="Path to workflow JSON configuration file")
    args = parser.parse_args()

    results = WorkflowRunner(args.config).run()
    if results.get("validation_summary", {}).get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
