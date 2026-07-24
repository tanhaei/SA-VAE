# Validation record

This repository was validated on 2026-07-24 with Python 3.12.13 on Linux.

## Command

```bash
bash scripts/smoke_test.sh
```

## Checks passed

- all Python source, test, and script files compiled;
- 11 unit/integration tests passed;
- patient-disjoint splitting and training-only categorical vocabularies passed;
- MCAR/MAR/MNAR masking invariants, weighted-MAR metadata, and probability
  calibration passed;
- cosine similarity and normalized neighbor-weight tests passed;
- the NumPy VAE trained and produced finite latent/reconstruction arrays;
- the end-to-end pipeline completed;
- 576 finite metric rows were generated for 8 methods;
- Accuracy, macro-Precision, macro-Recall, macro-F1, MAE, and RMSE were checked;
- 24 donor explanations had positive weights summing to one;
- 126 seed-level paired statistical comparisons had valid cluster-bootstrap
  intervals and raw/Holm-adjusted p-values;
- method lists agreed across metrics, efficiency data, and run metadata;
- partition record/patient counts agreed with cohort totals;
- every required CSV, JSON, YAML, and PNG artifact was present and non-empty.
- a clean ZIP extraction installed successfully in editable mode, the `savae`
  CLI validated the dataset, and the complete smoke suite passed again from
  that extracted copy.

The checked-in results under `results/smoke/` use 160 synthetic records from
80 artificial patients, with MCAR, MAR, and MNAR evaluation masks. They
establish software executability and internal consistency only; they do not
reproduce or validate the clinical performance claims reported for MIMIC-IV or
BioArc.
