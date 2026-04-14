#!/usr/bin/env python3
"""Entry point for standalone PyBullet validation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pybullet_validation import PyBulletValidationRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyBullet URDF torque consistency validation"
    )
    parser.add_argument("config", type=str, help="Path to validation JSON configuration")
    args = parser.parse_args()

    runner = PyBulletValidationRunner(args.config)
    summary = runner.run()
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
