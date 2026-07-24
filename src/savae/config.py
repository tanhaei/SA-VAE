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

    rate = float(config["evaluation"].get("mask_rate", -1))
    if not 0 < rate < 1:
        raise ConfigError("evaluation.mask_rate must be between 0 and 1.")

    repeats = int(config["evaluation"].get("repeats", 0))
    if repeats < 1:
        raise ConfigError("evaluation.repeats must be at least 1.")


def output_directory(config: dict[str, Any]) -> Path:
    path = Path(config["output"]["directory"])
    if not path.is_absolute():
        config_path = Path(config["_config_path"])
        path = (config_path.parent.parent / path).resolve()
    return path
