"""Leakage-safe baseline imputers used by the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import LabelEncoder

from .preprocessing import EncodedFrame, MixedTypePreprocessor
from .vae import NumpyVAE


class MeanModeImputer:
    name = "mean_mode"

    def __init__(self, preprocessor: MixedTypePreprocessor) -> None:
        self.preprocessor = preprocessor
        self.statistics: dict[str, object] = {}

    def fit(self, train_frame: pd.DataFrame, targets: list[str] | tuple[str, ...]) -> None:
        for target in targets:
            observed = train_frame[target].dropna()
            if observed.empty:
                raise ValueError(f"No observed training values for target {target}")
            if target in self.preprocessor.spec.continuous:
                self.statistics[target] = float(pd.to_numeric(observed).mean())
            else:
                self.statistics[target] = observed.astype(str).mode().iloc[0]

    def predict(self, n_rows: int, targets: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
        return {
            target: np.full(n_rows, self.statistics[target], dtype=object)
            for target in targets
        }


class RawKNNImputer:
    name = "knn"

    def __init__(self, preprocessor: MixedTypePreprocessor, k: int = 5) -> None:
        self.preprocessor = preprocessor
        self.k = int(k)

    def predict(
        self,
        train_frame: pd.DataFrame,
        train_encoded: EncodedFrame,
        query_encoded: EncodedFrame,
        targets: list[str] | tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        for target in targets:
            donor_mask = train_frame[target].notna().to_numpy()
            donor_positions = np.flatnonzero(donor_mask)
            if donor_positions.size == 0:
                raise ValueError(f"No observed donors for target {target}")
            donor_features = self.preprocessor.input_without_target(train_encoded, target)[
                donor_positions
            ]
            query_features = self.preprocessor.input_without_target(query_encoded, target)
            distances = _euclidean_distance_matrix(query_features, donor_features)
            top_local = np.argsort(distances, axis=1)[:, : min(self.k, len(donor_positions))]
            top_distances = np.take_along_axis(distances, top_local, axis=1)
            weights = 1.0 / (top_distances + 1e-6)
            weights /= weights.sum(axis=1, keepdims=True)
            selected = train_frame[target].to_numpy()[donor_positions[top_local]]

            if target in self.preprocessor.spec.continuous:
                predictions[target] = np.sum(weights * selected.astype(float), axis=1)
            else:
                categories = self.preprocessor.categories[target]
                scores = np.zeros((len(query_encoded.values), len(categories)))
                selected_strings = selected.astype(str)
                for category_index, category in enumerate(categories):
                    scores[:, category_index] = np.sum(
                        weights * (selected_strings == category), axis=1
                    )
                predictions[target] = np.asarray(categories, dtype=object)[
                    np.argmax(scores, axis=1)
                ]
        return predictions


class PlainVAEImputer:
    name = "plain_vae"

    def __init__(self, model: NumpyVAE, preprocessor: MixedTypePreprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor

    def predict(
        self,
        query_encoded: EncodedFrame,
        targets: list[str] | tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        decoded = self.model.reconstruct(query_encoded)
        return {
            target: self.preprocessor.decoder_target_to_raw(decoded, target)
            for target in targets
        }


@dataclass
class _FittedSupervisedTarget:
    model: object | None
    constant: object | None
    label_encoder: LabelEncoder | None


class SupervisedTreeImputer:
    """One leakage-safe supervised model per imputation target."""

    def __init__(
        self,
        preprocessor: MixedTypePreprocessor,
        kind: str,
        seed: int = 17,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.kind = kind
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.name = kind
        self.models: dict[str, _FittedSupervisedTarget] = {}

    def fit(
        self,
        train_frame: pd.DataFrame,
        train_encoded: EncodedFrame,
        targets: list[str] | tuple[str, ...],
    ) -> None:
        for target in targets:
            observed = train_frame[target].notna().to_numpy()
            features = self.preprocessor.input_without_target(train_encoded, target)[observed]
            raw_target = train_frame.loc[observed, target]
            if target in self.preprocessor.spec.continuous:
                values = pd.to_numeric(raw_target).to_numpy(dtype=float)
                if np.unique(values).size == 1:
                    self.models[target] = _FittedSupervisedTarget(None, float(values[0]), None)
                    continue
                model = self._regressor()
                model.fit(features, values)
                self.models[target] = _FittedSupervisedTarget(model, None, None)
            else:
                encoder = LabelEncoder()
                labels = encoder.fit_transform(raw_target.astype(str))
                if np.unique(labels).size == 1:
                    self.models[target] = _FittedSupervisedTarget(
                        None, str(raw_target.iloc[0]), encoder
                    )
                    continue
                model = self._classifier()
                model.fit(features, labels)
                self.models[target] = _FittedSupervisedTarget(model, None, encoder)

    def predict(
        self,
        query_encoded: EncodedFrame,
        targets: list[str] | tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        for target in targets:
            fitted = self.models[target]
            if fitted.model is None:
                predictions[target] = np.full(
                    len(query_encoded.values), fitted.constant, dtype=object
                )
                continue
            features = self.preprocessor.input_without_target(query_encoded, target)
            values = fitted.model.predict(features)
            if fitted.label_encoder is not None:
                values = fitted.label_encoder.inverse_transform(values.astype(int))
            predictions[target] = np.asarray(values)
        return predictions

    def _regressor(self):
        if self.kind == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=1,
            )
        if self.kind == "gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
            )
        if self.kind == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "XGBoost baseline requested but xgboost is not installed. "
                    "Install requirements-full.txt."
                ) from exc
            return XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth or 6,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=self.seed,
                n_jobs=1,
            )
        raise ValueError(f"Unknown supervised baseline: {self.kind}")

    def _classifier(self):
        if self.kind == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=1,
                class_weight="balanced",
            )
        if self.kind == "gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
            )
        if self.kind == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "XGBoost baseline requested but xgboost is not installed. "
                    "Install requirements-full.txt."
                ) from exc
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth or 6,
                learning_rate=0.05,
                objective="multi:softprob",
                random_state=self.seed,
                n_jobs=1,
            )
        raise ValueError(f"Unknown supervised baseline: {self.kind}")


def _euclidean_distance_matrix(queries: np.ndarray, donors: np.ndarray) -> np.ndarray:
    query_squared = np.sum(queries**2, axis=1, keepdims=True)
    donor_squared = np.sum(donors**2, axis=1, keepdims=True).T
    squared = np.maximum(query_squared + donor_squared - 2.0 * queries @ donors.T, 0.0)
    return np.sqrt(squared)

