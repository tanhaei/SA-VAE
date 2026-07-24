#!/usr/bin/env python3
"""Fail fast when a generated experiment directory is incomplete or inconsistent."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_FILES = (
    "cohort_summary.json",
    "config_resolved.yaml",
    "efficiency.csv",
    "efficiency_profile.png",
    "latent_space_pca.png",
    "masking_metadata.json",
    "metrics_long.csv",
    "metrics_summary.csv",
    "metrics_summary.png",
    "neighbor_explanations.json",
    "preprocessor_metadata.json",
    "run_metadata.json",
    "statistical_tests.csv",
    "training_history.csv",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AssertionError(f"{label} is non-finite: {value}")
    return parsed


def validate(
    output: Path,
    expected_metric_rows: int | None = None,
    expected_methods: set[str] | None = None,
) -> dict[str, object]:
    for filename in REQUIRED_FILES:
        path = output / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")

    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise AssertionError("run_metadata.json does not report a completed run")

    metrics = _read_csv(output / "metrics_long.csv")
    if expected_metric_rows is not None and len(metrics) != expected_metric_rows:
        raise AssertionError(
            f"Expected {expected_metric_rows} metric rows, found {len(metrics)}"
        )
    methods = {row["method"] for row in metrics}
    if expected_methods is not None and methods != expected_methods:
        raise AssertionError(
            f"Method mismatch: expected {sorted(expected_methods)}, found {sorted(methods)}"
        )
    for row_number, row in enumerate(metrics, start=2):
        value = _finite(row["value"], f"metrics_long.csv:{row_number}:value")
        if int(row["n_masked"]) <= 0:
            raise AssertionError(f"metrics_long.csv:{row_number}: n_masked must be positive")
        if row["metric"] in {
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        } and not 0.0 <= value <= 1.0:
            raise AssertionError(
                f"metrics_long.csv:{row_number}: classification metric outside [0, 1]"
            )
        if row["metric"] in {"mae", "rmse"} and value < 0.0:
            raise AssertionError(
                f"metrics_long.csv:{row_number}: error metric must be non-negative"
            )

    if int(metadata.get("metric_rows", -1)) != len(metrics):
        raise AssertionError("run metadata metric_rows does not match metrics_long.csv")
    if set(metadata.get("methods", [])) != methods:
        raise AssertionError("run metadata methods do not match metrics_long.csv")

    efficiency = _read_csv(output / "efficiency.csv")
    efficiency_methods = [row["method"] for row in efficiency]
    if len(efficiency_methods) != len(set(efficiency_methods)):
        raise AssertionError("efficiency.csv contains duplicate methods")
    if set(efficiency_methods) != methods:
        raise AssertionError("efficiency.csv methods do not match metrics_long.csv")
    for row_number, row in enumerate(efficiency, start=2):
        _finite(
            row["mean_inference_seconds_per_test_partition"],
            f"efficiency.csv:{row_number}:mean inference",
        )

    statistics = _read_csv(output / "statistical_tests.csv")
    for row_number, row in enumerate(statistics, start=2):
        for column in ("p_value", "holm_adjusted_p_value"):
            value = _finite(row[column], f"statistical_tests.csv:{row_number}:{column}")
            if not 0.0 <= value <= 1.0:
                raise AssertionError(
                    f"statistical_tests.csv:{row_number}:{column} outside [0, 1]"
                )

    explanations = json.loads(
        (output / "neighbor_explanations.json").read_text(encoding="utf-8")
    )
    if not explanations:
        raise AssertionError("neighbor_explanations.json contains no explanations")
    for position, explanation in enumerate(explanations):
        lengths = {
            len(explanation["donor_positions"]),
            len(explanation["similarities"]),
            len(explanation["weights"]),
            len(explanation["donor_values"]),
        }
        if len(lengths) != 1:
            raise AssertionError(f"Explanation {position} has inconsistent donor lengths")
        weights = [float(value) for value in explanation["weights"]]
        if any(not math.isfinite(value) or value <= 0 for value in weights):
            raise AssertionError(f"Explanation {position} has invalid weights")
        if not math.isclose(sum(weights), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"Explanation {position} weights do not sum to one")

    cohort = json.loads((output / "cohort_summary.json").read_text(encoding="utf-8"))
    partition_records = sum(
        int(value["records"]) for value in cohort["partitions"].values()
    )
    partition_patients = sum(
        int(value["patients"]) for value in cohort["partitions"].values()
    )
    if partition_records != int(cohort["total_records"]):
        raise AssertionError("Partition record counts do not sum to total_records")
    if partition_patients != int(cohort["total_patients"]):
        raise AssertionError(
            "Partition patient counts do not sum to total_patients; possible leakage"
        )

    return {
        "metric_rows": len(metrics),
        "methods": sorted(methods),
        "explanations": len(explanations),
        "statistical_comparisons": len(statistics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-metric-rows", type=int)
    parser.add_argument("--expected-methods", nargs="*")
    args = parser.parse_args()
    summary = validate(
        output=args.output.resolve(),
        expected_metric_rows=args.expected_metric_rows,
        expected_methods=set(args.expected_methods) if args.expected_methods else None,
    )
    print(
        "OUTPUT_VALIDATION_OK "
        f"metric_rows={summary['metric_rows']} "
        f"methods={len(summary['methods'])} "
        f"explanations={summary['explanations']} "
        f"statistical_comparisons={summary['statistical_comparisons']}"
    )


if __name__ == "__main__":
    main()
