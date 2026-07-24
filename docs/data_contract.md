# CSV data contract

The experiment runner expects one row per analysis unit (for example, a visit
or admission) and a stable patient identifier used only for group splitting.

## Required concepts

| Concept | Requirement |
|---|---|
| Patient ID | Non-missing; all rows for one patient share one value |
| Continuous fields | Parseable numeric values; missing values left blank/NA |
| Categorical fields | Strings or codes treated as nominal categories |
| Targets | Subset of continuous and categorical fields |
| Note text | Optional string column |

The exact column names are declared in the YAML configuration, not hard-coded.

## Example

```csv
patient_id,visit_id,age,iop,visual_acuity,diagnosis,followup_status,note_text
P00001,V001,61,23.4,0.40,glaucoma,urgent,"Elevated IOP; reduced vision."
P00002,V002,52,,0.10,normal,discharged,"Stable examination."
```

## Rules

1. Keep missing values missing. Do not pre-impute the CSV.
2. Do not one-hot encode or normalize before loading; the training-only
   preprocessor performs those steps.
3. Use a patient ID that prevents one person from crossing train/validation/test.
4. Remove direct identifiers and linkage keys before the research extract is
   created.
5. Keep protected extracts under `data/private/`, which is ignored by Git.
6. Record extraction dates, inclusion/exclusion criteria, units, coding
   systems, and natural missingness in the manuscript and a data dictionary.

## MIMIC-IV

Access and use remain governed by the PhysioNet credentialing and data-use
agreement. This repository does not download, redistribute, or infer access to
MIMIC-IV.

## BioArc

BioArc data are private operational records. Use requires the applicable ethics,
institutional, and data-governance approvals. This repository contains no
BioArc records.

