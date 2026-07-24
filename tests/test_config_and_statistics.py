from __future__ import annotations

import unittest

import pandas as pd

from savae.config import evaluation_mask_rates, evaluation_mask_seeds
from savae.statistics import paired_wilcoxon_table


class EvaluationConfigurationTests(unittest.TestCase):
    def test_explicit_rates_and_seeds_are_preserved(self) -> None:
        config = {
            "seed": 999,
            "evaluation": {
                "mask_rates": [0.10, 0.20, 0.30],
                "masking_seeds": [17, 29, 43],
            },
        }
        self.assertEqual(evaluation_mask_rates(config), [0.10, 0.20, 0.30])
        self.assertEqual(evaluation_mask_seeds(config), [17, 29, 43])


class StatisticalUnitTests(unittest.TestCase):
    def test_targets_are_aggregated_within_masking_seed(self) -> None:
        rows: list[dict] = []
        for repeat, seed in enumerate((17, 29, 43)):
            for target_index, target in enumerate(("diagnosis", "followup")):
                baseline_value = 0.60 + 0.01 * target_index
                proposed_value = baseline_value + 0.03 + 0.005 * repeat
                for method, value in (
                    ("baseline", baseline_value),
                    ("sa_vae", proposed_value),
                ):
                    rows.append(
                        {
                            "method": method,
                            "target": target,
                            "target_type": "categorical",
                            "metric": "accuracy",
                            "value": value,
                            "n_masked": 20,
                            "repeat": repeat,
                            "mask_seed": seed,
                            "mechanism": "mnar",
                            "mask_rate": 0.2,
                        }
                    )
        result = paired_wilcoxon_table(
            pd.DataFrame(rows),
            bootstrap_iterations=200,
            bootstrap_seed=5,
        )
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["paired_units"], 3)
        self.assertEqual(row["field_repeat_rows"], 6)
        self.assertEqual(row["unit_definition"], "masking_seed_mean_across_targets")
        self.assertLessEqual(row["bootstrap_ci_low"], row["bootstrap_ci_high"])


if __name__ == "__main__":
    unittest.main()

