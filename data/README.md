# Data directory

No MIMIC-IV or BioArc patient data are distributed in this repository.

- `data/raw/` and `data/private/` are ignored by Git.
- Use an authorized, de-identified extract that follows the contract in
  [`docs/data_contract.md`](../docs/data_contract.md).
- A synthetic ophthalmology-style dataset is generated in memory for the smoke
  test. It contains no real patient information.
- The unit used for splitting is `patient_id`; every record from one patient is
  kept in exactly one of train, validation, or test.

Never commit protected health information, MIMIC credentials, BioArc extracts,
or linkage keys.

