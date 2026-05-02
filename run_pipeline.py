#!/usr/bin/env python3
"""Entry point for the robot system identification pipeline."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import SystemIdentificationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Robot System Identification Pipeline"
    )
    parser.add_argument("config", type=str, help="Path to JSON configuration file")
    parser.add_argument(
        "--excitation-only", action="store_true", default=False,
        help="Stop after excitation optimisation (Stages 1-6) and save checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Resume from a checkpoint directory (skip Stages 5-6)",
    )
    args = parser.parse_args()

    pipeline = SystemIdentificationPipeline(args.config)
    if args.excitation_only:
        pipeline.cfg["excitation_only"] = True
    if args.checkpoint_dir:
        pipeline.cfg["checkpoint_dir"] = str(Path(args.checkpoint_dir).resolve())
    pipeline.run()


if __name__ == "__main__":
    main()
