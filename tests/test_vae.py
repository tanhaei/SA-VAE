from __future__ import annotations

import unittest

import numpy as np

from savae.data import DatasetSpec, patient_disjoint_split
from savae.preprocessing import MixedTypePreprocessor
from savae.synthetic import generate_synthetic_ehr
from savae.vae import NumpyVAE


class VaeTests(unittest.TestCase):
    def test_fit_encode_and_reconstruct_are_finite(self) -> None:
        frame = generate_synthetic_ehr(
            n_patients=35,
            visits_per_patient=1,
            natural_missing_rate=0.03,
            seed=13,
        )
        spec = DatasetSpec(
            patient_id="patient_id",
            continuous=("age", "iop", "visual_acuity"),
            categorical=("diagnosis", "sex"),
            targets=("iop", "diagnosis"),
        )
        train, validation, _ = patient_disjoint_split(
            frame,
            patient_column="patient_id",
            train_fraction=0.7,
            validation_fraction=0.15,
            test_fraction=0.15,
            seed=5,
        )
        processor = MixedTypePreprocessor(spec).fit(train)
        train_encoded = processor.transform(train)
        validation_encoded = processor.transform(validation)
        model = NumpyVAE(
            output_dimension=processor.output_dimension,
            text_dimension=0,
            continuous_indices=processor.continuous_indices,
            categorical_slices=processor.categorical_slices,
            hidden_dimension=10,
            latent_dimension=3,
            max_epochs=4,
            patience=2,
            batch_size=12,
            seed=8,
        ).fit(train_encoded, validation_encoded)
        latent = model.encode(validation_encoded)
        reconstruction = model.reconstruct(validation_encoded)
        self.assertEqual(latent.shape, (len(validation), 3))
        self.assertEqual(reconstruction.shape, validation_encoded.values.shape)
        self.assertTrue(np.isfinite(latent).all())
        self.assertTrue(np.isfinite(reconstruction).all())
        self.assertGreaterEqual(len(model.history), 1)
        self.assertGreater(model.parameter_count(), 0)


if __name__ == "__main__":
    unittest.main()

