from __future__ import annotations

import unittest

import numpy as np

from savae.missingness import apply_evaluation_mask
from savae.synthetic import generate_synthetic_ehr


class MissingnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = generate_synthetic_ehr(
            n_patients=50,
            visits_per_patient=1,
            natural_missing_rate=0.02,
            seed=4,
        )
        self.targets = ["iop", "diagnosis"]

    def test_masks_only_originally_observed_values(self) -> None:
        for mechanism in ("mcar", "mar", "mnar"):
            with self.subTest(mechanism=mechanism):
                result = apply_evaluation_mask(
                    self.frame,
                    self.targets,
                    mechanism=mechanism,
                    rate=0.2,
                    seed=9,
                    mar_drivers=["age", "diabetes"],
                )
                for target in self.targets:
                    selected = result.evaluation_mask[target].to_numpy()
                    self.assertGreater(selected.sum(), 0)
                    self.assertTrue(self.frame.loc[selected, target].notna().all())
                    self.assertTrue(result.masked_frame.loc[selected, target].isna().all())
                    unselected_observed = (~selected) & self.frame[target].notna().to_numpy()
                    original = self.frame.loc[unselected_observed, target].astype(str).to_numpy()
                    masked = (
                        result.masked_frame.loc[unselected_observed, target]
                        .astype(str)
                        .to_numpy()
                    )
                    np.testing.assert_array_equal(original, masked)

    def test_probability_calibration_is_close_to_requested_rate(self) -> None:
        result = apply_evaluation_mask(
            self.frame,
            ["iop"],
            mechanism="mnar",
            rate=0.25,
            seed=1,
        )
        observed = self.frame["iop"].notna()
        self.assertAlmostEqual(
            result.probabilities.loc[observed, "iop"].mean(),
            0.25,
            places=6,
        )

    def test_weighted_mar_records_declared_weights(self) -> None:
        result = apply_evaluation_mask(
            self.frame,
            ["iop"],
            mechanism="mar",
            rate=0.2,
            seed=3,
            mar_drivers=["age", "diabetes"],
            mar_driver_weights=[0.7, -0.3],
        )
        description = result.metadata["targets"]["iop"]["score"]
        self.assertIn("standardized weights", description)
        self.assertIn("-0.3", description)


if __name__ == "__main__":
    unittest.main()
