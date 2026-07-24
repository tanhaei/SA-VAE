"""A small, fully executable NumPy variational autoencoder.

The implementation is intentionally dependency-light so the repository's smoke
test can run without PyTorch. It uses a mask-aware mixed reconstruction loss,
the reparameterization trick, KL regularization, denoising corruption, Adam,
gradient clipping, and early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .preprocessing import EncodedFrame


@dataclass
class TrainingRecord:
    epoch: int
    train_loss: float
    validation_loss: float


class NumpyVAE:
    """One-hidden-layer mask-aware VAE for mixed encoded data."""

    def __init__(
        self,
        output_dimension: int,
        text_dimension: int,
        continuous_indices: Iterable[int],
        categorical_slices: Iterable[slice],
        hidden_dimension: int = 32,
        latent_dimension: int = 8,
        beta: float = 0.01,
        categorical_weight: float = 1.0,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 50,
        patience: int = 8,
        denoising_rate: float = 0.15,
        gradient_clip: float = 5.0,
        seed: int = 17,
    ) -> None:
        if output_dimension < 1 or latent_dimension < 1 or hidden_dimension < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= denoising_rate < 1:
            raise ValueError("denoising_rate must be in [0, 1)")
        if categorical_weight <= 0:
            raise ValueError("categorical_weight must be positive")
        self.output_dimension = int(output_dimension)
        self.text_dimension = int(text_dimension)
        self.input_dimension = 2 * self.output_dimension + self.text_dimension
        self.continuous_indices = tuple(int(index) for index in continuous_indices)
        self.categorical_slices = tuple(categorical_slices)
        self.hidden_dimension = int(hidden_dimension)
        self.latent_dimension = int(latent_dimension)
        self.beta = float(beta)
        self.categorical_weight = float(categorical_weight)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.denoising_rate = float(denoising_rate)
        self.gradient_clip = float(gradient_clip)
        self.seed = int(seed)
        self.history: list[TrainingRecord] = []
        self.best_epoch = 0
        self._rng = np.random.default_rng(self.seed)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        rng = self._rng

        def xavier(rows: int, columns: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (rows + columns))
            return rng.uniform(-limit, limit, size=(rows, columns)).astype(np.float64)

        self.parameters = {
            "encoder_weight": xavier(self.input_dimension, self.hidden_dimension),
            "encoder_bias": np.zeros(self.hidden_dimension, dtype=np.float64),
            "mu_weight": xavier(self.hidden_dimension, self.latent_dimension),
            "mu_bias": np.zeros(self.latent_dimension, dtype=np.float64),
            "logvar_weight": xavier(self.hidden_dimension, self.latent_dimension),
            "logvar_bias": np.zeros(self.latent_dimension, dtype=np.float64),
            "decoder_weight": xavier(self.latent_dimension, self.hidden_dimension),
            "decoder_bias": np.zeros(self.hidden_dimension, dtype=np.float64),
            "output_weight": xavier(self.hidden_dimension, self.output_dimension),
            "output_bias": np.zeros(self.output_dimension, dtype=np.float64),
        }
        self._adam_m = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_v = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self._adam_step = 0

    def fit(self, train: EncodedFrame, validation: EncodedFrame) -> "NumpyVAE":
        self._validate_encoded(train)
        self._validate_encoded(validation)
        best_parameters = self._copy_parameters()
        best_loss = np.inf
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(1, self.max_epochs + 1):
            permutation = self._rng.permutation(len(train.values))
            batch_losses: list[float] = []
            for start in range(0, len(permutation), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                input_matrix = self._corrupted_input(
                    train.values[indices],
                    train.observed_mask[indices],
                    train.text[indices],
                    self._rng,
                )
                loss, gradients = self._loss_and_gradients(
                    input_matrix,
                    train.values[indices],
                    train.observed_mask[indices],
                )
                gradients = self._clip_gradients(gradients)
                self._adam_update(gradients)
                batch_losses.append(loss)

            validation_rng = np.random.default_rng(self.seed + 10_000 + epoch)
            validation_input = self._corrupted_input(
                validation.values,
                validation.observed_mask,
                validation.text,
                validation_rng,
            )
            validation_loss = self._loss_only(
                validation_input,
                validation.values,
                validation.observed_mask,
            )
            train_loss = float(np.mean(batch_losses))
            self.history.append(
                TrainingRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                )
            )

            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_parameters = self._copy_parameters()
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        self.parameters = best_parameters
        return self

    def encode(self, encoded: EncodedFrame) -> np.ndarray:
        self._validate_encoded(encoded)
        model_input = encoded.model_input()
        hidden = np.tanh(
            model_input @ self.parameters["encoder_weight"]
            + self.parameters["encoder_bias"]
        )
        return hidden @ self.parameters["mu_weight"] + self.parameters["mu_bias"]

    def reconstruct(self, encoded: EncodedFrame) -> np.ndarray:
        latent_mean = self.encode(encoded)
        return self.decode(latent_mean)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        hidden = np.tanh(
            latent @ self.parameters["decoder_weight"] + self.parameters["decoder_bias"]
        )
        logits = hidden @ self.parameters["output_weight"] + self.parameters["output_bias"]
        decoded = logits.copy()
        for group_slice in self.categorical_slices:
            decoded[:, group_slice] = _softmax(logits[:, group_slice])
        return decoded

    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.parameters.values()))

    def history_as_dicts(self) -> list[dict]:
        return [
            {
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "validation_loss": record.validation_loss,
            }
            for record in self.history
        ]

    def _loss_and_gradients(
        self,
        model_input: np.ndarray,
        target_values: np.ndarray,
        target_mask: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        p = self.parameters
        encoder_pre = model_input @ p["encoder_weight"] + p["encoder_bias"]
        encoder_hidden = np.tanh(encoder_pre)
        mu = encoder_hidden @ p["mu_weight"] + p["mu_bias"]
        raw_logvar = encoder_hidden @ p["logvar_weight"] + p["logvar_bias"]
        logvar = np.clip(raw_logvar, -8.0, 8.0)
        logvar_derivative = ((raw_logvar > -8.0) & (raw_logvar < 8.0)).astype(np.float64)
        standard_deviation = np.exp(0.5 * logvar)
        epsilon = self._rng.normal(size=mu.shape)
        latent = mu + standard_deviation * epsilon
        decoder_pre = latent @ p["decoder_weight"] + p["decoder_bias"]
        decoder_hidden = np.tanh(decoder_pre)
        logits = decoder_hidden @ p["output_weight"] + p["output_bias"]

        reconstruction_loss, output_gradient = self._reconstruction_loss_and_gradient(
            logits, target_values, target_mask
        )
        kl_loss = -0.5 * np.mean(
            np.sum(1.0 + logvar - mu * mu - np.exp(logvar), axis=1)
        )
        total_loss = reconstruction_loss + self.beta * kl_loss

        gradients: dict[str, np.ndarray] = {}
        gradients["output_weight"] = decoder_hidden.T @ output_gradient
        gradients["output_bias"] = output_gradient.sum(axis=0)
        decoder_hidden_gradient = output_gradient @ p["output_weight"].T
        decoder_pre_gradient = decoder_hidden_gradient * (1.0 - decoder_hidden**2)
        gradients["decoder_weight"] = latent.T @ decoder_pre_gradient
        gradients["decoder_bias"] = decoder_pre_gradient.sum(axis=0)
        latent_gradient = decoder_pre_gradient @ p["decoder_weight"].T

        batch_size = max(len(model_input), 1)
        mu_gradient = latent_gradient + self.beta * mu / batch_size
        logvar_gradient = (
            latent_gradient * (0.5 * standard_deviation * epsilon)
            + self.beta * 0.5 * (np.exp(logvar) - 1.0) / batch_size
        )
        logvar_gradient *= logvar_derivative

        gradients["mu_weight"] = encoder_hidden.T @ mu_gradient
        gradients["mu_bias"] = mu_gradient.sum(axis=0)
        gradients["logvar_weight"] = encoder_hidden.T @ logvar_gradient
        gradients["logvar_bias"] = logvar_gradient.sum(axis=0)

        encoder_hidden_gradient = (
            mu_gradient @ p["mu_weight"].T
            + logvar_gradient @ p["logvar_weight"].T
        )
        encoder_pre_gradient = encoder_hidden_gradient * (1.0 - encoder_hidden**2)
        gradients["encoder_weight"] = model_input.T @ encoder_pre_gradient
        gradients["encoder_bias"] = encoder_pre_gradient.sum(axis=0)
        return float(total_loss), gradients

    def _loss_only(
        self,
        model_input: np.ndarray,
        target_values: np.ndarray,
        target_mask: np.ndarray,
    ) -> float:
        p = self.parameters
        encoder_hidden = np.tanh(
            model_input @ p["encoder_weight"] + p["encoder_bias"]
        )
        mu = encoder_hidden @ p["mu_weight"] + p["mu_bias"]
        logvar = np.clip(
            encoder_hidden @ p["logvar_weight"] + p["logvar_bias"],
            -8.0,
            8.0,
        )
        decoder_hidden = np.tanh(mu @ p["decoder_weight"] + p["decoder_bias"])
        logits = decoder_hidden @ p["output_weight"] + p["output_bias"]
        reconstruction_loss, _ = self._reconstruction_loss_and_gradient(
            logits, target_values, target_mask
        )
        kl_loss = -0.5 * np.mean(
            np.sum(1.0 + logvar - mu * mu - np.exp(logvar), axis=1)
        )
        return float(reconstruction_loss + self.beta * kl_loss)

    def _reconstruction_loss_and_gradient(
        self,
        logits: np.ndarray,
        target_values: np.ndarray,
        target_mask: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        output_gradient = np.zeros_like(logits)
        loss_sum = 0.0
        observed_units = 0.0

        for index in self.continuous_indices:
            mask = target_mask[:, index]
            difference = logits[:, index] - target_values[:, index]
            loss_sum += 0.5 * float(np.sum(mask * difference**2))
            output_gradient[:, index] = mask * difference
            observed_units += float(mask.sum())

        for group_slice in self.categorical_slices:
            row_mask = target_mask[:, group_slice.start]
            probabilities = _softmax(logits[:, group_slice])
            targets = target_values[:, group_slice]
            loss_sum -= self.categorical_weight * float(
                np.sum(row_mask[:, None] * targets * np.log(probabilities + 1e-12))
            )
            output_gradient[:, group_slice] = self.categorical_weight * (
                row_mask[:, None] * (probabilities - targets)
            )
            observed_units += self.categorical_weight * float(row_mask.sum())

        denominator = max(observed_units, 1.0)
        return loss_sum / denominator, output_gradient / denominator

    def _corrupted_input(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        text: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        input_mask = observed_mask.copy()
        if self.denoising_rate > 0:
            for index in self.continuous_indices:
                observed = input_mask[:, index] > 0
                drop = observed & (rng.random(len(values)) < self.denoising_rate)
                input_mask[drop, index] = 0.0
            for group_slice in self.categorical_slices:
                observed = input_mask[:, group_slice.start] > 0
                drop = observed & (rng.random(len(values)) < self.denoising_rate)
                input_mask[drop, group_slice] = 0.0
        input_values = values * input_mask
        return np.concatenate([input_values, input_mask, text], axis=1)

    def _clip_gradients(self, gradients: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        norm = np.sqrt(sum(float(np.sum(value * value)) for value in gradients.values()))
        if not np.isfinite(norm):
            raise FloatingPointError("Non-finite gradient norm")
        if norm <= self.gradient_clip or norm == 0:
            return gradients
        scale = self.gradient_clip / norm
        return {name: value * scale for name, value in gradients.items()}

    def _adam_update(self, gradients: dict[str, np.ndarray]) -> None:
        self._adam_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name, gradient in gradients.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1 - beta1) * gradient
            self._adam_v[name] = beta2 * self._adam_v[name] + (1 - beta2) * gradient**2
            corrected_m = self._adam_m[name] / (1 - beta1**self._adam_step)
            corrected_v = self._adam_v[name] / (1 - beta2**self._adam_step)
            self.parameters[name] -= (
                self.learning_rate * corrected_m / (np.sqrt(corrected_v) + epsilon)
            )

    def _copy_parameters(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.parameters.items()}

    def _validate_encoded(self, encoded: EncodedFrame) -> None:
        if encoded.values.shape != encoded.observed_mask.shape:
            raise ValueError("values and observed_mask shapes differ")
        if encoded.values.shape[1] != self.output_dimension:
            raise ValueError("Encoded output dimension does not match the model")
        if encoded.text.shape != (len(encoded.values), self.text_dimension):
            raise ValueError("Encoded text dimension does not match the model")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
