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
    args = parser.parse_args()

    pipeline = SystemIdentificationPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
