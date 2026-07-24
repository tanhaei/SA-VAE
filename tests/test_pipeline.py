from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from savae.experiment import run_experiment


class PipelineTests(unittest.TestCase):
    def test_tiny_end_to_end_run_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result"
            config = {
                "seed": 22,
                "_config_path": str(Path(temporary_directory) / "config.yaml"),
                "data": {
                    "source": "synthetic",
                    "synthetic": {
                        "n_patients": 40,
                        "visits_per_patient": 1,
                        "natural_missing_rate": 0.02,
                    },
                    "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
                },
                "columns": {
                    "patient_id": "patient_id",
                    "note_text": "note_text",
                    "continuous": ["age", "iop", "visual_acuity"],
                    "categorical": ["sex", "diagnosis", "followup_status"],
                    "targets": ["iop", "diagnosis"],
                },
                "text": {"backend": "hash", "dimension": 6},
                "model": {
                    "hidden_dimension": 10,
                    "latent_dimension": 3,
                    "max_epochs": 3,
                    "patience": 2,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "beta": 0.01,
                    "denoising_rate": 0.2,
                },
                "neighbors": {"k": 3, "temperature": 0.3},
                "baselines": {"knn_k": 3},
                "ablation": {"structured_only": False},
                "evaluation": {
                    "mechanisms": ["mcar"],
                    "mask_rate": 0.25,
                    "repeats": 1,
                    "baselines": ["mean_mode", "knn"],
                    "explain_limit": 1,
                },
                "output": {"directory": str(output)},
            }
            artifacts = run_experiment(config)
            self.assertFalse(artifacts.metrics.empty)
            self.assertIn("sa_vae", set(artifacts.metrics["method"]))
            for filename in (
                "metrics_long.csv",
                "metrics_summary.csv",
                "run_metadata.json",
                "neighbor_explanations.json",
                "metrics_summary.png",
            ):
                self.assertTrue((output / filename).is_file(), filename)
                self.assertGreater((output / filename).stat().st_size, 0)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")


if __name__ == "__main__":
    unittest.main()

