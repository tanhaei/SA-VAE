"""Dataset loading, schema validation, and patient-disjoint splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .synthetic import generate_synthetic_ehr


@dataclass(frozen=True)
class DatasetSpec:
    patient_id: str
    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    targets: tuple[str, ...]
    note_text: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DatasetSpec":
        columns = config["columns"]
        return cls(
            patient_id=str(columns["patient_id"]),
            continuous=tuple(columns["continuous"]),
            categorical=tuple(columns["categorical"]),
            targets=tuple(columns["targets"]),
            note_text=columns.get("note_text"),
        )

    @property
    def features(self) -> tuple[str, ...]:
        return self.continuous + self.categorical

    def validate(self, frame: pd.DataFrame) -> None:
        required = {self.patient_id, *self.features}
        if self.note_text:
            required.add(self.note_text)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        if frame[self.patient_id].isna().any():
            raise ValueError("patient_id cannot contain missing values")
        if frame[self.patient_id].nunique() < 10:
            raise ValueError("At least 10 distinct patients are required")


def load_dataset(config: dict[str, Any], spec: DatasetSpec) -> pd.DataFrame:
    data_config = config["data"]
    source = str(data_config.get("source", "csv")).lower()
    if source == "synthetic":
        synthetic = data_config.get("synthetic", {})
        frame = generate_synthetic_ehr(
            n_patients=int(synthetic.get("n_patients", 160)),
            visits_per_patient=int(synthetic.get("visits_per_patient", 2)),
            natural_missing_rate=float(synthetic.get("natural_missing_rate", 0.04)),
            seed=int(config["seed"]),
        )
    elif source == "csv":
        path = Path(data_config["path"])
        if not path.exists():
            raise FileNotFoundError(f"CSV dataset not found: {path}")
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data.source: {source}")

    max_rows = data_config.get("max_rows")
    if max_rows:
        frame = frame.iloc[: int(max_rows)].copy()
    spec.validate(frame)
    return frame.reset_index(drop=True)


def patient_disjoint_split(
    frame: pd.DataFrame,
    patient_column: str,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split records by patient; no patient can occur in two partitions."""

    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to one")

    groups = frame[patient_column].astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(outer.split(frame, groups=groups))
    train_val = frame.iloc[train_val_idx].copy()
    test = frame.iloc[test_idx].copy()

    relative_validation = validation_fraction / (train_fraction + validation_fraction)
    inner_groups = train_val[patient_column].astype(str).to_numpy()
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation,
        random_state=seed + 1,
    )
    train_idx, validation_idx = next(inner.split(train_val, groups=inner_groups))
    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[validation_idx].copy()

    partitions = [
        set(train[patient_column].astype(str)),
        set(validation[patient_column].astype(str)),
        set(test[patient_column].astype(str)),
    ]
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise AssertionError("Patient leakage detected across partitions")

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
