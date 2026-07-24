"""Artificial evaluation masking with explicit MCAR, MAR, and MNAR mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MaskingResult:
    masked_frame: pd.DataFrame
    evaluation_mask: pd.DataFrame
    probabilities: pd.DataFrame
    metadata: dict


def apply_evaluation_mask(
    frame: pd.DataFrame,
    targets: list[str] | tuple[str, ...],
    mechanism: str,
    rate: float,
    seed: int,
    mar_drivers: list[str] | tuple[str, ...] | None = None,
    strength: float = 1.25,
) -> MaskingResult:
    """Hide originally observed target entries without destroying ground truth.

    ``evaluation_mask.loc[i, target]`` is true exactly when an originally
    observed test value is hidden and should be scored.
    """

    if not 0 < rate < 1:
        raise ValueError("rate must be between zero and one")
    mechanism = mechanism.lower()
    if mechanism not in {"mcar", "mar", "mnar"}:
        raise ValueError("mechanism must be one of: mcar, mar, mnar")

    rng = np.random.default_rng(seed)
    masked = frame.copy()
    evaluation_mask = pd.DataFrame(False, index=frame.index, columns=list(targets))
    probabilities = pd.DataFrame(0.0, index=frame.index, columns=list(targets))
    details: dict[str, dict] = {}

    for target in targets:
        observed = frame[target].notna().to_numpy()
        eligible_indices = np.flatnonzero(observed)
        if eligible_indices.size < 2:
            raise ValueError(f"Target {target!r} has fewer than two observed test values")

        if mechanism == "mcar":
            eligible_probabilities = np.full(eligible_indices.size, rate, dtype=np.float64)
            score_description = "constant probability"
        elif mechanism == "mar":
            drivers = [column for column in (mar_drivers or []) if column != target]
            if not drivers:
                drivers = [column for column in frame.columns if column != target][:2]
            scores = _combined_observed_driver_score(frame.loc[:, drivers])
            eligible_scores = scores[eligible_indices]
            eligible_probabilities = _calibrated_probabilities(
                eligible_scores, target_rate=rate, strength=strength
            )
            score_description = f"observed drivers: {drivers}"
        else:
            target_scores = _target_value_score(frame[target])
            eligible_scores = target_scores[eligible_indices]
            eligible_probabilities = _calibrated_probabilities(
                eligible_scores, target_rate=rate, strength=strength
            )
            score_description = "value-dependent self-masking"

        random_values = rng.random(eligible_indices.size)
        hidden_local = random_values < eligible_probabilities
        if hidden_local.sum() == 0:
            hidden_local[np.argmax(eligible_probabilities - random_values)] = True
        if hidden_local.sum() == eligible_indices.size:
            hidden_local[np.argmin(eligible_probabilities - random_values)] = False

        hidden_indices = eligible_indices[hidden_local]
        evaluation_mask.iloc[hidden_indices, evaluation_mask.columns.get_loc(target)] = True
        probabilities.iloc[eligible_indices, probabilities.columns.get_loc(target)] = (
            eligible_probabilities
        )
        masked.iloc[hidden_indices, masked.columns.get_loc(target)] = np.nan
        details[target] = {
            "eligible": int(eligible_indices.size),
            "hidden": int(hidden_indices.size),
            "realized_rate": float(hidden_indices.size / eligible_indices.size),
            "mean_probability": float(eligible_probabilities.mean()),
            "score": score_description,
        }

    return MaskingResult(
        masked_frame=masked,
        evaluation_mask=evaluation_mask,
        probabilities=probabilities,
        metadata={
            "mechanism": mechanism,
            "requested_rate": float(rate),
            "seed": int(seed),
            "strength": float(strength),
            "targets": details,
        },
    )


def _combined_observed_driver_score(drivers: pd.DataFrame) -> np.ndarray:
    components: list[np.ndarray] = []
    for column in drivers.columns:
        series = drivers[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(2, len(series) // 3):
            values = numeric.to_numpy(dtype=np.float64, copy=True)
            finite = np.isfinite(values)
            median = float(np.nanmedian(values)) if finite.any() else 0.0
            values[~finite] = median
        else:
            observed = series.dropna().astype(str)
            levels = {value: index for index, value in enumerate(sorted(set(observed)))}
            values = np.array(
                [levels.get(str(value), 0) if pd.notna(value) else 0 for value in series],
                dtype=np.float64,
            )
        components.append(_standardize(values))
    if not components:
        return np.zeros(len(drivers), dtype=np.float64)
    return np.mean(np.column_stack(components), axis=1)


def _target_value_score(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(2, series.notna().sum() // 2):
        values = numeric.to_numpy(dtype=np.float64, copy=True)
        finite = np.isfinite(values)
        fill = float(np.nanmedian(values)) if finite.any() else 0.0
        values[~finite] = fill
        return _standardize(values)

    observed = series.dropna().astype(str)
    frequencies = observed.value_counts(normalize=True).to_dict()
    scores = np.array(
        [
            -np.log(max(frequencies.get(str(value), 1.0), 1e-12)) if pd.notna(value) else 0.0
            for value in series
        ],
        dtype=np.float64,
    )
    return _standardize(scores)


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def _calibrated_probabilities(
    scores: np.ndarray,
    target_rate: float,
    strength: float,
) -> np.ndarray:
    if scores.size == 0:
        return np.empty(0, dtype=np.float64)
    lower, upper = -30.0, 30.0
    for _ in range(100):
        intercept = (lower + upper) / 2
        probabilities = _sigmoid(intercept + strength * scores)
        if probabilities.mean() < target_rate:
            lower = intercept
        else:
            upper = intercept
    return _sigmoid((lower + upper) / 2 + strength * scores)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))
