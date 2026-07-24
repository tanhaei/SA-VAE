#!/usr/bin/env python3
"""Run an experiment without requiring package installation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from savae.config import load_config  # noqa: E402
from savae.experiment import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    artifacts = run_experiment(load_config(args.config))
    print(f"SMOKE_OK output={artifacts.output_directory}")
    print(f"metrics={len(artifacts.metrics)} methods={artifacts.metrics['method'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

