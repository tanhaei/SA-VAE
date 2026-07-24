"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("The YAML root must be a mapping.")

    config["_config_path"] = str(config_path)
    _validate(config)

    data_path = config.get("data", {}).get("path")
    if data_path:
        path_obj = Path(data_path)
        if not path_obj.is_absolute():
            config["data"]["path"] = str((config_path.parent / path_obj).resolve())
    return config


def _validate(config: dict[str, Any]) -> None:
    for key in ("seed", "data", "columns", "model", "evaluation", "output"):
        if key not in config:
            raise ConfigError(f"Missing required top-level key: {key}")

    columns = config["columns"]
    for key in ("patient_id", "continuous", "categorical", "targets"):
        if key not in columns:
            raise ConfigError(f"Missing columns.{key}")

    continuous = set(columns["continuous"])
    categorical = set(columns["categorical"])
    targets = set(columns["targets"])
    if continuous & categorical:
        raise ConfigError("Continuous and categorical columns must be disjoint.")
    if not targets:
        raise ConfigError("At least one target column is required.")
    if not targets <= continuous | categorical:
        missing = sorted(targets - (continuous | categorical))
        raise ConfigError(f"Targets absent from feature definitions: {missing}")

    split = config["data"].get("split", {})
    fractions = [float(split.get(name, 0.0)) for name in ("train", "validation", "test")]
    if abs(sum(fractions) - 1.0) > 1e-8 or any(value <= 0 for value in fractions):
        raise ConfigError(
            "data.split train/validation/test fractions must be positive and sum to 1."
        )

    evaluation = config["evaluation"]
    has_single_rate = "mask_rate" in evaluation
    has_multiple_rates = "mask_rates" in evaluation
    if has_single_rate == has_multiple_rates:
        raise ConfigError(
            "Specify exactly one of evaluation.mask_rate or evaluation.mask_rates."
        )
    rates = evaluation_mask_rates(config)
    if any(not 0 < rate < 1 for rate in rates):
        raise ConfigError("Every evaluation masking rate must be between 0 and 1.")
    if len(set(rates)) != len(rates):
        raise ConfigError("evaluation.mask_rates must not contain duplicates.")

    has_repeats = "repeats" in evaluation
    has_masking_seeds = "masking_seeds" in evaluation
    if not has_repeats and not has_masking_seeds:
        raise ConfigError(
            "Specify evaluation.repeats or an explicit evaluation.masking_seeds list."
        )
    if has_repeats and int(evaluation["repeats"]) < 1:
        raise ConfigError("evaluation.repeats must be at least 1.")
    if has_masking_seeds:
        seeds = evaluation["masking_seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError("evaluation.masking_seeds must be a non-empty list.")
        parsed_seeds = [int(value) for value in seeds]
        if len(set(parsed_seeds)) != len(parsed_seeds):
            raise ConfigError("evaluation.masking_seeds must not contain duplicates.")
        if has_repeats and int(evaluation["repeats"]) != len(parsed_seeds):
            raise ConfigError(
                "evaluation.repeats must equal the length of evaluation.masking_seeds."
            )

    drivers = list(evaluation.get("mar_drivers", []))
    weights = evaluation.get("mar_driver_weights")
    if weights is not None:
        if not isinstance(weights, list):
            raise ConfigError("evaluation.mar_driver_weights must be a list.")
        if len(weights) != len(drivers):
            raise ConfigError(
                "evaluation.mar_driver_weights must match evaluation.mar_drivers."
            )
        parsed_weights = [float(value) for value in weights]
        if not any(abs(value) > 0 for value in parsed_weights):
            raise ConfigError("At least one MAR driver weight must be non-zero.")

    bootstrap_iterations = int(evaluation.get("bootstrap_iterations", 20_000))
    if bootstrap_iterations < 1:
        raise ConfigError("evaluation.bootstrap_iterations must be at least 1.")


def output_directory(config: dict[str, Any]) -> Path:
    path = Path(config["output"]["directory"])
    if not path.is_absolute():
        config_path = Path(config["_config_path"])
        path = (config_path.parent.parent / path).resolve()
    return path


def evaluation_mask_rates(config: dict[str, Any]) -> list[float]:
    """Return one or more declared artificial-masking rates."""

    evaluation = config["evaluation"]
    if "mask_rates" in evaluation:
        values = evaluation["mask_rates"]
        if not isinstance(values, list) or not values:
            raise ConfigError("evaluation.mask_rates must be a non-empty list.")
        return [float(value) for value in values]
    return [float(evaluation["mask_rate"])]


def evaluation_mask_seeds(config: dict[str, Any]) -> list[int]:
    """Return explicit masking seeds, deriving them only for legacy configs."""

    evaluation = config["evaluation"]
    if "masking_seeds" in evaluation:
        return [int(value) for value in evaluation["masking_seeds"]]
    repeats = int(evaluation["repeats"])
    base_seed = int(config["seed"])
    return [base_seed + repeat for repeat in range(repeats)]
