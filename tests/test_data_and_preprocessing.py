from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from savae.data import DatasetSpec, patient_disjoint_split
from savae.preprocessing import MixedTypePreprocessor
from savae.synthetic import generate_synthetic_ehr


class DataAndPreprocessingTests(unittest.TestCase):
    def test_patient_split_has_no_overlap(self) -> None:
        frame = generate_synthetic_ehr(
            n_patients=40,
            visits_per_patient=2,
            natural_missing_rate=0.0,
            seed=3,
        )
        train, validation, test = patient_disjoint_split(
            frame,
            patient_column="patient_id",
            train_fraction=0.7,
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=7,
        )
        train_patients = set(train["patient_id"])
        validation_patients = set(validation["patient_id"])
        test_patients = set(test["patient_id"])
        self.assertFalse(train_patients & validation_patients)
        self.assertFalse(train_patients & test_patients)
        self.assertFalse(validation_patients & test_patients)

    def test_unknown_category_does_not_change_training_vocabulary(self) -> None:
        spec = DatasetSpec(
            patient_id="patient_id",
            continuous=("age",),
            categorical=("diagnosis",),
            targets=("diagnosis",),
        )
        train = pd.DataFrame(
            {
                "patient_id": ["a", "b", "c"],
                "age": [20.0, 30.0, 40.0],
                "diagnosis": ["normal", "cataract", "normal"],
            }
        )
        test = pd.DataFrame(
            {
                "patient_id": ["d"],
                "age": [50.0],
                "diagnosis": ["unseen_diagnosis"],
            }
        )
        processor = MixedTypePreprocessor(spec).fit(train)
        encoded = processor.transform(test)
        group = processor.feature_slices["diagnosis"]
        self.assertEqual(encoded.values[0, group].sum(), 1.0)
        unknown_position = processor.categories["diagnosis"].index(processor.unknown_token)
        self.assertEqual(encoded.values[0, group.start + unknown_position], 1.0)
        self.assertTrue(np.all(encoded.observed_mask[0, group] == 1.0))


if __name__ == "__main__":
    unittest.main()

