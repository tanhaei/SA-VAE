"""Paired Wilcoxon comparisons with Holm multiplicity correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LOWER_IS_BETTER = {"mae", "rmse"}


def paired_wilcoxon_table(
    metrics: pd.DataFrame,
    proposed_method: str = "sa_vae",
) -> pd.DataFrame:
    rows: list[dict] = []
    methods = sorted(set(metrics["method"]) - {proposed_method})
    grouping_columns = ["mechanism", "mask_rate", "metric", "target_type"]

    for group_key, group in metrics.groupby(grouping_columns, dropna=False):
        proposed = group[group["method"] == proposed_method]
        if proposed.empty:
            continue
        raw_rows: list[dict] = []
        for baseline in methods:
            baseline_rows = group[group["method"] == baseline]
            if baseline_rows.empty:
                continue
            paired = proposed.merge(
                baseline_rows,
                on=["target", "repeat"],
                suffixes=("_proposed", "_baseline"),
            )
            if paired.empty:
                continue
            metric = str(group_key[2])
            if metric in LOWER_IS_BETTER:
                difference = (
                    paired["value_baseline"].to_numpy()
                    - paired["value_proposed"].to_numpy()
                )
            else:
                difference = (
                    paired["value_proposed"].to_numpy()
                    - paired["value_baseline"].to_numpy()
                )
            nonzero = difference[np.abs(difference) > 1e-12]
            if nonzero.size == 0:
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    difference,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            raw_rows.append(
                {
                    "baseline": baseline,
                    "proposed": proposed_method,
                    "mechanism": group_key[0],
                    "mask_rate": group_key[1],
                    "metric": metric,
                    "target_type": group_key[3],
                    "paired_units": int(len(paired)),
                    "median_effect_favoring_proposed": float(np.median(difference)),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
        adjusted = _holm_adjust([row["p_value"] for row in raw_rows])
        for row, adjusted_p in zip(raw_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            rows.append(row)

    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty_like(values)
    running_max = 0.0
    n_values = len(values)
    for rank, ordered_index in enumerate(order):
        adjusted_value = min((n_values - rank) * values[ordered_index], 1.0)
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[ordered_index] = running_max
    return adjusted_sorted.astype(float).tolist()

