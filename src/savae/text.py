"""Text encoders used to augment the structured patient representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextEncoder(Protocol):
    dimension: int

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Encode texts as a dense floating-point matrix."""


@dataclass
class HashingTextEncoder:
    """Dependency-light deterministic encoder used by tests and smoke runs.

    This is not ClinicalBERT. It provides a reproducible substitute when model
    weights are unavailable and makes that substitution explicit in metadata.
    """

    dimension: int = 16
    ngram_range: tuple[int, int] = (1, 2)

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2",
            ngram_range=self.ngram_range,
        )

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        normalized = ["" if text is None else str(text) for text in texts]
        return self._vectorizer.transform(normalized).toarray().astype(np.float64)


class ClinicalBERTTextEncoder:
    """Optional frozen ClinicalBERT encoder.

    Install ``requirements-full.txt`` and provide a model ID in the YAML
    configuration. No network call is hidden by this class: Hugging Face handles
    local cache lookup/download according to the caller's environment.
    """

    def __init__(
        self,
        model_id: str,
        pooling: str = "mean",
        max_length: int = 256,
        batch_size: int = 8,
        device: str = "cpu",
        long_note_strategy: str = "truncate",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ClinicalBERT requires torch and transformers. "
                "Install with: pip install -r requirements-full.txt"
            ) from exc

        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        if long_note_strategy not in {"truncate", "segment_mean"}:
            raise ValueError(
                "long_note_strategy must be 'truncate' or 'segment_mean'"
            )
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device)
        self._model.eval()
        self.pooling = pooling
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.long_note_strategy = long_note_strategy
        self.dimension = int(self._model.config.hidden_size)

    def transform(self, texts: Iterable[str]) -> np.ndarray:  # pragma: no cover - optional
        values = ["" if text is None else str(text) for text in texts]
        if self.long_note_strategy == "segment_mean":
            return self._transform_segmented(values)

        chunks: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            chunks.append(self._pool_token_batch(tokens))
        return np.vstack(chunks) if chunks else np.empty((0, self.dimension))

    def _transform_segmented(self, values: list[str]) -> np.ndarray:
        note_vectors: list[np.ndarray] = []
        for text in values:
            tokens = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            tokens.pop("overflow_to_sample_mapping", None)
            segment_vectors: list[np.ndarray] = []
            segment_count = int(tokens["input_ids"].shape[0])
            for start in range(0, segment_count, self.batch_size):
                batch_tokens = {
                    key: value[start : start + self.batch_size]
                    for key, value in tokens.items()
                }
                segment_vectors.append(self._pool_token_batch(batch_tokens))
            note_vectors.append(np.vstack(segment_vectors).mean(axis=0))
        return (
            np.vstack(note_vectors)
            if note_vectors
            else np.empty((0, self.dimension), dtype=np.float64)
        )

    def _pool_token_batch(self, tokens) -> np.ndarray:
        torch = self._torch
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = self._model(**tokens).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                attention = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(
                    min=1
                )
        return pooled.cpu().numpy().astype(np.float64)


def build_text_encoder(config: dict) -> TextEncoder | None:
    backend = str(config.get("backend", "none")).lower()
    if backend == "none":
        return None
    if backend == "hash":
        return HashingTextEncoder(dimension=int(config.get("dimension", 16)))
    if backend == "clinicalbert":
        model_id = config.get("model_id")
        if not model_id:
            raise ValueError("text.backend=clinicalbert requires text.model_id")
        return ClinicalBERTTextEncoder(
            model_id=model_id,
            pooling=str(config.get("pooling", "mean")),
            max_length=int(config.get("max_length", 256)),
            batch_size=int(config.get("batch_size", 8)),
            device=str(config.get("device", "cpu")),
            long_note_strategy=str(config.get("long_note_strategy", "truncate")),
        )
    raise ValueError(f"Unknown text backend: {backend}")
