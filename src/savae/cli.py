"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import DatasetSpec, load_dataset
from .experiment import run_experiment
from .synthetic import generate_synthetic_ehr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="savae",
        description="Similarity-augmented VAE experiments for EHR imputation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from YAML")
    run_parser.add_argument("--config", required=True, type=Path)

    validate_parser = subparsers.add_parser("validate", help="Validate config and dataset")
    validate_parser.add_argument("--config", required=True, type=Path)

    synthetic_parser = subparsers.add_parser(
        "generate-synthetic", help="Write a non-clinical example CSV"
    )
    synthetic_parser.add_argument("--output", required=True, type=Path)
    synthetic_parser.add_argument("--patients", type=int, default=160)
    synthetic_parser.add_argument("--visits-per-patient", type=int, default=2)
    synthetic_parser.add_argument("--missing-rate", type=float, default=0.04)
    synthetic_parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        artifacts = run_experiment(config)
        print(f"Experiment completed: {artifacts.output_directory}")
        print(f"Metric rows: {len(artifacts.metrics)}")
        if artifacts.skipped_methods:
            print(f"Skipped methods: {sorted(artifacts.skipped_methods)}")
        return 0

    if args.command == "validate":
        config = load_config(args.config)
        spec = DatasetSpec.from_config(config)
        frame = load_dataset(config, spec)
        print(
            f"Valid dataset: {len(frame)} records, "
            f"{frame[spec.patient_id].nunique()} patients"
        )
        return 0

    if args.command == "generate-synthetic":
        frame = generate_synthetic_ehr(
            n_patients=args.patients,
            visits_per_patient=args.visits_per_patient,
            natural_missing_rate=args.missing_rate,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Wrote {len(frame)} synthetic rows to {args.output}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

