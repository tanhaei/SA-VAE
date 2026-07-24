"""Field-level metrics with categorical and continuous outcomes kept separate."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from .data import DatasetSpec


def score_predictions(
    truth: pd.DataFrame,
    evaluation_mask: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    spec: DatasetSpec,
    repeat: int,
    mask_seed: int,
    mechanism: str,
    mask_rate: float,
) -> list[dict]:
    rows: list[dict] = []
    for method, method_predictions in predictions.items():
        for target in spec.targets:
            selected = evaluation_mask[target].to_numpy(dtype=bool)
            n_selected = int(selected.sum())
            if n_selected == 0:
                continue
            y_true = truth.loc[selected, target].to_numpy()
            y_pred = np.asarray(method_predictions[target])[selected]
            if target in spec.continuous:
                y_true_numeric = y_true.astype(float)
                y_pred_numeric = y_pred.astype(float)
                metric_values = {
                    "mae": float(mean_absolute_error(y_true_numeric, y_pred_numeric)),
                    "rmse": float(
                        np.sqrt(mean_squared_error(y_true_numeric, y_pred_numeric))
                    ),
                }
                target_type = "continuous"
            else:
                y_true_string = y_true.astype(str)
                y_pred_string = y_pred.astype(str)
                metric_values = {
                    "accuracy": float(accuracy_score(y_true_string, y_pred_string)),
                    "macro_precision": float(
                        precision_score(
                            y_true_string,
                            y_pred_string,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "macro_recall": float(
                        recall_score(
                            y_true_string,
                            y_pred_string,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "macro_f1": float(
                        f1_score(
                            y_true_string,
                            y_pred_string,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
                target_type = "categorical"

            for metric, value in metric_values.items():
                rows.append(
                    {
                        "method": method,
                        "target": target,
                        "target_type": target_type,
                        "metric": metric,
                        "value": value,
                        "n_masked": n_selected,
                        "repeat": int(repeat),
                        "mask_seed": int(mask_seed),
                        "mechanism": mechanism,
                        "mask_rate": float(mask_rate),
                    }
                )
    return rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(
            ["method", "target_type", "metric", "mechanism", "mask_rate"],
            dropna=False,
        )
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            median=("value", "median"),
            minimum=("value", "min"),
            maximum=("value", "max"),
            evaluation_units=("value", "size"),
            masked_entries=("n_masked", "sum"),
        )
        .reset_index()
        .sort_values(["target_type", "metric", "mean", "method"])
    )
