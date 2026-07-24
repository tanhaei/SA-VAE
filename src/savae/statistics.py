"""Paired Wilcoxon comparisons with Holm multiplicity correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LOWER_IS_BETTER = {"mae", "rmse"}


def paired_wilcoxon_table(
    metrics: pd.DataFrame,
    proposed_method: str = "sa_vae",
    bootstrap_iterations: int = 20_000,
    bootstrap_seed: int = 17,
) -> pd.DataFrame:
    """Compare methods using masking seeds as the independent paired units.

    Field-level differences from the same masking seed are averaged before the
    Wilcoxon test. The confidence interval resamples those seed-level effects,
    so multiple targets evaluated on the same held-out patients are not treated
    as independent replicates.
    """

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1")
    rows: list[dict] = []
    methods = sorted(set(metrics["method"]) - {proposed_method})
    grouping_columns = ["mechanism", "mask_rate", "metric", "target_type"]
    rng = np.random.default_rng(bootstrap_seed)

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
                on=["target", "repeat", "mask_seed"],
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
            paired = paired.assign(effect=difference)
            seed_effects = (
                paired.groupby(["repeat", "mask_seed"], dropna=False)["effect"]
                .mean()
                .to_numpy(dtype=float)
            )
            nonzero = seed_effects[np.abs(seed_effects) > 1e-12]
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
            interval_low, interval_high = _cluster_bootstrap_median_interval(
                seed_effects,
                iterations=bootstrap_iterations,
                rng=rng,
            )
            raw_rows.append(
                {
                    "baseline": baseline,
                    "proposed": proposed_method,
                    "mechanism": group_key[0],
                    "mask_rate": group_key[1],
                    "metric": metric,
                    "target_type": group_key[3],
                    "unit_definition": "masking_seed_mean_across_targets",
                    "paired_units": int(len(seed_effects)),
                    "field_repeat_rows": int(len(paired)),
                    "median_effect_favoring_proposed": float(
                        np.median(seed_effects)
                    ),
                    "bootstrap_ci_low": interval_low,
                    "bootstrap_ci_high": interval_high,
                    "bootstrap_iterations": int(bootstrap_iterations),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
        adjusted = _holm_adjust([row["p_value"] for row in raw_rows])
        for row, adjusted_p in zip(raw_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            rows.append(row)

    return pd.DataFrame(rows)


def _cluster_bootstrap_median_interval(
    seed_effects: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(seed_effects, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    sampled_indices = rng.integers(
        low=0,
        high=values.size,
        size=(iterations, values.size),
    )
    bootstrapped = np.median(values[sampled_indices], axis=1)
    low, high = np.quantile(bootstrapped, [0.025, 0.975])
    return float(low), float(high)


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
