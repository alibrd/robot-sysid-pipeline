#!/usr/bin/env python3
"""Unified entry point: run pipeline, validation, report, benchmark, and/or plot from one JSON."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.runner import UnifiedRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robot system identification — unified entry point"
    )
    parser.add_argument("config", type=str, help="Path to unified JSON configuration")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Run only this stage: excitation|identification|"
            "validation|report|benchmark|plot"
        ),
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Disable a stage (can be repeated)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a previous output directory (overrides top-level 'checkpoint'). "
            "Relative paths are resolved against the current working directory, "
            "not the config file's location."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the merged config; do not execute any stage",
    )
    args = parser.parse_args()

    runner = UnifiedRunner(args.config)
    if args.only:
        runner.set_only(args.only)
    for stage in args.skip:
        runner.disable_stage(stage)
    if args.resume:
        runner.set_resume(args.resume)

    if args.dry_run:
        runner.print_resolved_config()
        return 0

    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
