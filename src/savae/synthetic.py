"""Synthetic mixed-type EHR generator for smoke tests and examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONDITIONS = np.array(["normal", "cataract", "glaucoma", "diabetic_retinopathy"])


def generate_synthetic_ehr(
    n_patients: int = 160,
    visits_per_patient: int = 2,
    natural_missing_rate: float = 0.04,
    seed: int = 17,
) -> pd.DataFrame:
    """Generate correlated ophthalmology-style records without real patient data."""

    if n_patients < 20:
        raise ValueError("n_patients must be at least 20 for stable group splits")
    if visits_per_patient < 1:
        raise ValueError("visits_per_patient must be positive")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    condition_probs = np.array([0.28, 0.30, 0.24, 0.18])
    condition_iop = {
        "normal": 14.5,
        "cataract": 15.5,
        "glaucoma": 23.0,
        "diabetic_retinopathy": 17.5,
    }
    condition_va = {
        "normal": 0.08,
        "cataract": 0.55,
        "glaucoma": 0.35,
        "diabetic_retinopathy": 0.62,
    }
    medications = {
        "normal": ["none", "lubricant"],
        "cataract": ["none", "steroid"],
        "glaucoma": ["timolol", "latanoprost"],
        "diabetic_retinopathy": ["anti_vegf", "none"],
    }

    for patient_index in range(n_patients):
        patient_id = f"P{patient_index:05d}"
        condition = str(rng.choice(CONDITIONS, p=condition_probs))
        base_age = float(np.clip(rng.normal(57 if condition != "normal" else 45, 13), 18, 92))
        sex = str(rng.choice(["female", "male"]))
        diabetes = int(condition == "diabetic_retinopathy" or rng.random() < 0.14)

        for visit_index in range(visits_per_patient):
            visit_noise = rng.normal(0, 1)
            iop = condition_iop[condition] + 0.8 * visit_noise + rng.normal(0, 1.5)
            visual_acuity = max(
                0.0,
                condition_va[condition] + 0.025 * visit_index + rng.normal(0, 0.12),
            )
            medication = str(rng.choice(medications[condition]))
            surgery = "yes" if condition == "cataract" and rng.random() < 0.65 else "no"
            followup_status = (
                "urgent"
                if iop > 24 or visual_acuity > 0.75
                else ("routine" if condition != "normal" else "discharged")
            )
            note = (
                f"{condition.replace('_', ' ')} follow-up; "
                f"IOP {'elevated' if iop > 21 else 'stable'}; "
                f"vision {'reduced' if visual_acuity > 0.4 else 'preserved'}; "
                f"medication {medication}."
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "visit_id": f"{patient_id}-V{visit_index + 1}",
                    "age": base_age + visit_index / 2,
                    "iop": float(iop),
                    "visual_acuity": float(visual_acuity),
                    "followup_days": float(
                        np.clip(120 - 3.2 * iop - 35 * visual_acuity + rng.normal(0, 12), 3, 180)
                    ),
                    "sex": sex,
                    "diabetes": str(diabetes),
                    "diagnosis": condition,
                    "medication": medication,
                    "surgery": surgery,
                    "followup_status": followup_status,
                    "note_text": note,
                }
            )

    frame = pd.DataFrame(rows)
    feature_columns = [
        "age",
        "iop",
        "visual_acuity",
        "followup_days",
        "sex",
        "diabetes",
        "diagnosis",
        "medication",
        "surgery",
        "followup_status",
    ]
    for column in feature_columns:
        missing = rng.random(len(frame)) < natural_missing_rate
        # Keep enough observations in every target and class for the smoke test.
        if missing.sum() > len(frame) // 4:
            missing[:] = False
        frame.loc[missing, column] = np.nan

    return frame

