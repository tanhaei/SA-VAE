"""Field-specific latent-neighbor imputation and donor explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocessing import EncodedFrame, MixedTypePreprocessor
from .vae import NumpyVAE


@dataclass
class NeighborExplanation:
    query_position: int
    target: str
    donor_positions: list[int]
    similarities: list[float]
    weights: list[float]
    donor_values: list[object]

    def as_dict(self) -> dict:
        return {
            "query_position": self.query_position,
            "target": self.target,
            "donor_positions": self.donor_positions,
            "similarities": self.similarities,
            "weights": self.weights,
            "donor_values": self.donor_values,
        }


class LatentNeighborImputer:
    """Impute with normalized top-k weights in a VAE latent space."""

    def __init__(
        self,
        model: NumpyVAE,
        preprocessor: MixedTypePreprocessor,
        k: int = 5,
        temperature: float = 0.2,
        epsilon: float = 1e-8,
    ) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.preprocessor = preprocessor
        self.k = int(k)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)

    def predict(
        self,
        train_frame: pd.DataFrame,
        train_encoded: EncodedFrame,
        query_encoded: EncodedFrame,
        targets: list[str] | tuple[str, ...],
        explain_limit: int = 3,
    ) -> tuple[dict[str, np.ndarray], list[NeighborExplanation]]:
        predictions: dict[str, np.ndarray] = {}
        explanations: list[NeighborExplanation] = []
        fallback_decoded = self.model.reconstruct(query_encoded)

        for target in targets:
            donor_observed = train_frame[target].notna().to_numpy()
            donor_positions = np.flatnonzero(donor_observed)
            if donor_positions.size == 0:
                predictions[target] = self.preprocessor.decoder_target_to_raw(
                    fallback_decoded, target
                )
                continue

            donor_encoded = _mask_field(train_encoded, self.preprocessor, target)
            query_target_masked = _mask_field(query_encoded, self.preprocessor, target)
            donor_latent = self.model.encode(donor_encoded)[donor_positions]
            query_latent = self.model.encode(query_target_masked)
            similarity = cosine_similarity_matrix(
                query_latent, donor_latent, epsilon=self.epsilon
            )
            top_local = np.argsort(-similarity, axis=1)[:, : min(self.k, len(donor_positions))]
            top_similarity = np.take_along_axis(similarity, top_local, axis=1)
            weights = normalized_similarity_weights(top_similarity, self.temperature)
            selected_donor_positions = donor_positions[top_local]
            donor_values = train_frame[target].to_numpy()[selected_donor_positions]

            if target in self.preprocessor.spec.continuous:
                numeric_values = donor_values.astype(np.float64)
                target_prediction = np.sum(weights * numeric_values, axis=1)
            else:
                categories = self.preprocessor.categories[target]
                category_scores = np.zeros((len(query_encoded.values), len(categories)))
                for category_index, category in enumerate(categories):
                    category_scores[:, category_index] = np.sum(
                        weights * (donor_values.astype(str) == category),
                        axis=1,
                    )
                target_prediction = np.asarray(categories, dtype=object)[
                    np.argmax(category_scores, axis=1)
                ]
            predictions[target] = target_prediction

            for query_position in range(min(explain_limit, len(query_encoded.values))):
                explanations.append(
                    NeighborExplanation(
                        query_position=query_position,
                        target=target,
                        donor_positions=(
                            selected_donor_positions[query_position]
                            .astype(int)
                            .tolist()
                        ),
                        similarities=top_similarity[query_position].astype(float).tolist(),
                        weights=weights[query_position].astype(float).tolist(),
                        donor_values=[
                            _python_scalar(value) for value in donor_values[query_position].tolist()
                        ],
                    )
                )

        return predictions, explanations


def cosine_similarity_matrix(
    queries: np.ndarray,
    donors: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    donor_norm = np.linalg.norm(donors, axis=1, keepdims=True).T
    return (queries @ donors.T) / (query_norm * donor_norm + epsilon)


def normalized_similarity_weights(similarities: np.ndarray, temperature: float) -> np.ndarray:
    scaled = similarities / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(scaled, -60.0, 60.0))
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def _mask_field(
    encoded: EncodedFrame,
    preprocessor: MixedTypePreprocessor,
    target: str,
) -> EncodedFrame:
    values = encoded.values.copy()
    observed_mask = encoded.observed_mask.copy()
    indices = preprocessor.target_indices(target)
    values[:, indices] = 0.0
    observed_mask[:, indices] = 0.0
    return EncodedFrame(
        values=values,
        observed_mask=observed_mask,
        text=encoded.text,
        index=encoded.index,
    )


def _python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value
