"""Leakage-safe preprocessing for mixed continuous and categorical EHR fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DatasetSpec
from .text import TextEncoder


@dataclass
class EncodedFrame:
    values: np.ndarray
    observed_mask: np.ndarray
    text: np.ndarray
    index: pd.Index

    def model_input(self) -> np.ndarray:
        return np.concatenate([self.values, self.observed_mask, self.text], axis=1)


class MixedTypePreprocessor:
    """Fit normalization/categories on training data and transform every split."""

    unknown_token = "__UNK__"

    def __init__(self, spec: DatasetSpec, text_encoder: TextEncoder | None = None) -> None:
        self.spec = spec
        self.text_encoder = text_encoder
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, tuple[str, ...]] = {}
        self.feature_slices: dict[str, slice] = {}
        self.continuous_indices: list[int] = []
        self.categorical_slices: list[slice] = []
        self._fitted = False
        self.output_dimension = 0
        self.text_dimension = int(text_encoder.dimension) if text_encoder else 0

    def fit(self, frame: pd.DataFrame) -> "MixedTypePreprocessor":
        cursor = 0
        for column in self.spec.continuous:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            mean = float(numeric.mean())
            std = float(numeric.std(ddof=0))
            if not np.isfinite(mean):
                raise ValueError(f"Continuous training column has no observed values: {column}")
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.means[column] = mean
            self.stds[column] = std
            self.feature_slices[column] = slice(cursor, cursor + 1)
            self.continuous_indices.append(cursor)
            cursor += 1

        for column in self.spec.categorical:
            observed = frame[column].dropna().astype(str)
            levels = sorted(set(observed))
            if not levels:
                raise ValueError(f"Categorical training column has no observed values: {column}")
            if self.unknown_token not in levels:
                levels.append(self.unknown_token)
            categories = tuple(levels)
            self.categories[column] = categories
            group_slice = slice(cursor, cursor + len(categories))
            self.feature_slices[column] = group_slice
            self.categorical_slices.append(group_slice)
            cursor += len(categories)

        self.output_dimension = cursor
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> EncodedFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        n_rows = len(frame)
        values = np.zeros((n_rows, self.output_dimension), dtype=np.float64)
        observed_mask = np.zeros_like(values)

        for column in self.spec.continuous:
            index = self.feature_slices[column].start
            numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            observed = np.isfinite(numeric)
            values[observed, index] = (
                numeric[observed] - self.means[column]
            ) / self.stds[column]
            observed_mask[observed, index] = 1.0

        for column in self.spec.categorical:
            group_slice = self.feature_slices[column]
            categories = self.categories[column]
            category_to_index = {value: idx for idx, value in enumerate(categories)}
            raw = frame[column]
            observed = raw.notna().to_numpy()
            for row_index in np.flatnonzero(observed):
                value = str(raw.iloc[row_index])
                local_index = category_to_index.get(value, category_to_index[self.unknown_token])
                values[row_index, group_slice.start + local_index] = 1.0
                observed_mask[row_index, group_slice] = 1.0

        if self.text_encoder and self.spec.note_text:
            text = self.text_encoder.transform(frame[self.spec.note_text].fillna("").astype(str))
        else:
            text = np.empty((n_rows, 0), dtype=np.float64)
        return EncodedFrame(
            values=values,
            observed_mask=observed_mask,
            text=text,
            index=frame.index,
        )

    def target_indices(self, column: str) -> np.ndarray:
        group_slice = self.feature_slices[column]
        return np.arange(group_slice.start, group_slice.stop)

    def input_without_target(self, encoded: EncodedFrame, column: str) -> np.ndarray:
        values = encoded.values.copy()
        mask = encoded.observed_mask.copy()
        indices = self.target_indices(column)
        values[:, indices] = 0.0
        mask[:, indices] = 0.0
        return np.concatenate([values, mask, encoded.text], axis=1)

    def decoder_target_to_raw(self, decoded: np.ndarray, column: str) -> np.ndarray:
        group_slice = self.feature_slices[column]
        if column in self.spec.continuous:
            standardized = decoded[:, group_slice.start]
            return standardized * self.stds[column] + self.means[column]

        probabilities = decoded[:, group_slice]
        class_indices = np.argmax(probabilities, axis=1)
        categories = np.asarray(self.categories[column], dtype=object)
        return categories[class_indices]

    def field_metadata(self) -> dict:
        return {
            "output_dimension": self.output_dimension,
            "text_dimension": self.text_dimension,
            "continuous_indices": list(self.continuous_indices),
            "categorical_slices": [
                [group_slice.start, group_slice.stop] for group_slice in self.categorical_slices
            ],
            "feature_slices": {
                name: [group_slice.start, group_slice.stop]
                for name, group_slice in self.feature_slices.items()
            },
            "categories": {name: list(values) for name, values in self.categories.items()},
            "means": self.means,
            "stds": self.stds,
        }

