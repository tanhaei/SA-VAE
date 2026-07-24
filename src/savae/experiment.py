"""End-to-end experiment orchestration and artifact generation."""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from sklearn.decomposition import PCA

from .baselines import (
    MeanModeImputer,
    PlainVAEImputer,
    RawKNNImputer,
    SupervisedTreeImputer,
)
from .config import evaluation_mask_rates, evaluation_mask_seeds, output_directory
from .data import DatasetSpec, load_dataset, patient_disjoint_split
from .metrics import score_predictions, summarize_metrics
from .missingness import apply_evaluation_mask
from .preprocessing import MixedTypePreprocessor
from .similarity import LatentNeighborImputer
from .statistics import paired_wilcoxon_table
from .text import build_text_encoder
from .vae import NumpyVAE


@dataclass
class ExperimentArtifacts:
    output_directory: Path
    metrics: pd.DataFrame
    summary: pd.DataFrame
    statistics: pd.DataFrame
    efficiency: pd.DataFrame
    skipped_methods: dict[str, str]


def run_experiment(config: dict[str, Any]) -> ExperimentArtifacts:
    started = time.perf_counter()
    seed = int(config["seed"])
    np.random.seed(seed)
    spec = DatasetSpec.from_config(config)
    frame = load_dataset(config, spec)

    split_config = config["data"]["split"]
    train, validation, test = patient_disjoint_split(
        frame=frame,
        patient_column=spec.patient_id,
        train_fraction=float(split_config["train"]),
        validation_fraction=float(split_config["validation"]),
        test_fraction=float(split_config["test"]),
        seed=seed,
    )

    text_encoder = build_text_encoder(config.get("text", {"backend": "none"}))
    preprocessor = MixedTypePreprocessor(spec, text_encoder=text_encoder).fit(train)
    train_encoded = preprocessor.transform(train)
    validation_encoded = preprocessor.transform(validation)
    test_encoded = preprocessor.transform(test)

    training_efficiency: dict[str, dict[str, float | int | str | None]] = {}
    model = _build_vae(config["model"], preprocessor, seed=seed)
    train_start = time.perf_counter()
    model.fit(train_encoded, validation_encoded)
    shared_vae_training_seconds = time.perf_counter() - train_start
    for method_name in ("plain_vae", "sa_vae"):
        training_efficiency[method_name] = {
            "training_seconds": shared_vae_training_seconds,
            "parameter_count": model.parameter_count(),
            "training_note": "shared VAE training; methods differ only at imputation",
        }

    structured_preprocessor: MixedTypePreprocessor | None = None
    structured_train_encoded = None
    structured_validation_encoded = None
    structured_model: NumpyVAE | None = None
    if bool(config.get("ablation", {}).get("structured_only", False)):
        structured_preprocessor = MixedTypePreprocessor(spec, text_encoder=None).fit(train)
        structured_train_encoded = structured_preprocessor.transform(train)
        structured_validation_encoded = structured_preprocessor.transform(validation)
        structured_model = _build_vae(config["model"], structured_preprocessor, seed=seed + 73)
        train_start = time.perf_counter()
        structured_model.fit(structured_train_encoded, structured_validation_encoded)
        shared_structured_training_seconds = time.perf_counter() - train_start
        for method_name in ("plain_vae_structured", "sa_vae_structured"):
            training_efficiency[method_name] = {
                "training_seconds": shared_structured_training_seconds,
                "parameter_count": structured_model.parameter_count(),
                "training_note": "shared structured-only VAE training",
            }

    targets = list(spec.targets)
    fitted_baselines: dict[str, object] = {}
    skipped_methods: dict[str, str] = {}
    baseline_names = list(config["evaluation"].get("baselines", []))
    baseline_config = config.get("baselines", {})

    if "mean_mode" in baseline_names:
        start = time.perf_counter()
        baseline = MeanModeImputer(preprocessor)
        baseline.fit(train, targets)
        fitted_baselines["mean_mode"] = baseline
        training_efficiency["mean_mode"] = {
            "training_seconds": time.perf_counter() - start,
            "parameter_count": None,
            "training_note": "summary-statistic fit",
        }
    if "knn" in baseline_names:
        fitted_baselines["knn"] = RawKNNImputer(
            preprocessor, k=int(baseline_config.get("knn_k", 5))
        )
        training_efficiency["knn"] = {
            "training_seconds": 0.0,
            "parameter_count": None,
            "training_note": "lazy donor lookup",
        }

    for tree_name in ("random_forest", "gradient_boosting", "xgboost"):
        if tree_name not in baseline_names:
            continue
        tree_config = baseline_config.get(tree_name, {})
        baseline = SupervisedTreeImputer(
            preprocessor=preprocessor,
            kind=tree_name,
            seed=seed,
            n_estimators=int(tree_config.get("n_estimators", 100)),
            max_depth=tree_config.get("max_depth"),
            learning_rate=float(tree_config.get("learning_rate", 0.05)),
            subsample=float(tree_config.get("subsample", 1.0)),
            colsample_bytree=float(tree_config.get("colsample_bytree", 1.0)),
        )
        start = time.perf_counter()
        try:
            baseline.fit(train, train_encoded, targets)
        except RuntimeError as exc:
            if bool(config["evaluation"].get("skip_unavailable_baselines", False)):
                skipped_methods[tree_name] = str(exc)
                continue
            raise
        fitted_baselines[tree_name] = baseline
        training_efficiency[tree_name] = {
            "training_seconds": time.perf_counter() - start,
            "parameter_count": None,
            "training_note": "one supervised model per target",
        }

    metric_rows: list[dict] = []
    masking_metadata: list[dict] = []
    explanation_records: list[dict] = []
    prediction_timings: dict[str, list[float]] = {}
    evaluation = config["evaluation"]
    mechanisms = list(evaluation.get("mechanisms", [evaluation.get("mechanism", "mcar")]))
    mask_rates = evaluation_mask_rates(config)
    masking_seeds = evaluation_mask_seeds(config)
    mar_drivers = list(evaluation.get("mar_drivers", []))
    mar_driver_weights = evaluation.get("mar_driver_weights")
    strength = float(evaluation.get("masking_strength", 1.25))
    neighbor_config = config.get("neighbors", {})

    for mechanism in mechanisms:
        for mask_rate in mask_rates:
            for repeat, mask_seed in enumerate(masking_seeds):
                masking = apply_evaluation_mask(
                    frame=test,
                    targets=targets,
                    mechanism=mechanism,
                    rate=mask_rate,
                    seed=mask_seed,
                    mar_drivers=mar_drivers,
                    mar_driver_weights=mar_driver_weights,
                    strength=strength,
                )
                masking_metadata.append(masking.metadata)
                query_encoded = preprocessor.transform(masking.masked_frame)
                predictions: dict[str, dict[str, np.ndarray]] = {}

                for baseline_name, baseline in fitted_baselines.items():
                    start = time.perf_counter()
                    if baseline_name == "mean_mode":
                        method_predictions = baseline.predict(len(test), targets)
                    elif baseline_name == "knn":
                        method_predictions = baseline.predict(
                            train, train_encoded, query_encoded, targets
                        )
                    else:
                        method_predictions = baseline.predict(query_encoded, targets)
                    prediction_timings.setdefault(baseline_name, []).append(
                        time.perf_counter() - start
                    )
                    predictions[baseline_name] = method_predictions

                plain_vae = PlainVAEImputer(model, preprocessor)
                start = time.perf_counter()
                predictions["plain_vae"] = plain_vae.predict(query_encoded, targets)
                prediction_timings.setdefault("plain_vae", []).append(
                    time.perf_counter() - start
                )

                latent_imputer = LatentNeighborImputer(
                    model=model,
                    preprocessor=preprocessor,
                    k=int(neighbor_config.get("k", 5)),
                    temperature=float(neighbor_config.get("temperature", 0.2)),
                )
                start = time.perf_counter()
                sa_predictions, explanations = latent_imputer.predict(
                    train_frame=train,
                    train_encoded=train_encoded,
                    query_encoded=query_encoded,
                    targets=targets,
                    explain_limit=(
                        int(evaluation.get("explain_limit", 3)) if repeat == 0 else 0
                    ),
                )
                prediction_timings.setdefault("sa_vae", []).append(
                    time.perf_counter() - start
                )
                predictions["sa_vae"] = sa_predictions
                if repeat == 0:
                    explanation_records.extend(
                        {
                            **explanation.as_dict(),
                            "mechanism": mechanism,
                            "mask_rate": float(mask_rate),
                            "mask_seed": int(mask_seed),
                            "repeat": repeat,
                        }
                        for explanation in explanations
                    )

                if (
                    structured_model is not None
                    and structured_preprocessor is not None
                    and structured_train_encoded is not None
                ):
                    structured_query = structured_preprocessor.transform(
                        masking.masked_frame
                    )
                    start = time.perf_counter()
                    predictions["plain_vae_structured"] = PlainVAEImputer(
                        structured_model, structured_preprocessor
                    ).predict(structured_query, targets)
                    prediction_timings.setdefault("plain_vae_structured", []).append(
                        time.perf_counter() - start
                    )

                    structured_similarity = LatentNeighborImputer(
                        model=structured_model,
                        preprocessor=structured_preprocessor,
                        k=int(neighbor_config.get("k", 5)),
                        temperature=float(neighbor_config.get("temperature", 0.2)),
                    )
                    start = time.perf_counter()
                    structured_predictions, _ = structured_similarity.predict(
                        train_frame=train,
                        train_encoded=structured_train_encoded,
                        query_encoded=structured_query,
                        targets=targets,
                        explain_limit=0,
                    )
                    prediction_timings.setdefault("sa_vae_structured", []).append(
                        time.perf_counter() - start
                    )
                    predictions["sa_vae_structured"] = structured_predictions

                metric_rows.extend(
                    score_predictions(
                        truth=test,
                        evaluation_mask=masking.evaluation_mask,
                        predictions=predictions,
                        spec=spec,
                        repeat=repeat,
                        mask_seed=mask_seed,
                        mechanism=mechanism,
                        mask_rate=mask_rate,
                    )
                )

    efficiency_rows: list[dict] = []
    for method in sorted(set(training_efficiency) | set(prediction_timings)):
        training = training_efficiency.get(method, {})
        durations = prediction_timings.get(method, [])
        efficiency_rows.append(
            {
                "method": method,
                "training_seconds": training.get("training_seconds", np.nan),
                "mean_inference_seconds_per_test_partition": (
                    float(np.mean(durations)) if durations else np.nan
                ),
                "median_inference_seconds_per_test_partition": (
                    float(np.median(durations)) if durations else np.nan
                ),
                "parameter_count": training.get("parameter_count", np.nan),
                "process_peak_rss_mb": _peak_rss_mb(),
                "training_note": training.get("training_note", ""),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    statistics = paired_wilcoxon_table(
        metrics,
        proposed_method="sa_vae",
        bootstrap_iterations=int(evaluation.get("bootstrap_iterations", 20_000)),
        bootstrap_seed=seed,
    )
    efficiency = pd.DataFrame(efficiency_rows)

    output_dir = output_directory(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_artifacts(
        output_dir=output_dir,
        config=config,
        frame=frame,
        train=train,
        validation=validation,
        test=test,
        spec=spec,
        preprocessor=preprocessor,
        model=model,
        structured_model=structured_model,
        test_encoded=test_encoded,
        metrics=metrics,
        summary=summary,
        statistics=statistics,
        efficiency=efficiency,
        masking_metadata=masking_metadata,
        explanation_records=explanation_records,
        skipped_methods=skipped_methods,
        elapsed_seconds=time.perf_counter() - started,
    )
    return ExperimentArtifacts(
        output_directory=output_dir,
        metrics=metrics,
        summary=summary,
        statistics=statistics,
        efficiency=efficiency,
        skipped_methods=skipped_methods,
    )


def _build_vae(
    model_config: dict[str, Any],
    preprocessor: MixedTypePreprocessor,
    seed: int,
) -> NumpyVAE:
    return NumpyVAE(
        output_dimension=preprocessor.output_dimension,
        text_dimension=preprocessor.text_dimension,
        continuous_indices=preprocessor.continuous_indices,
        categorical_slices=preprocessor.categorical_slices,
        hidden_dimension=int(model_config.get("hidden_dimension", 32)),
        latent_dimension=int(model_config.get("latent_dimension", 8)),
        beta=float(model_config.get("beta", 0.01)),
        categorical_weight=float(model_config.get("categorical_weight", 1.0)),
        learning_rate=float(model_config.get("learning_rate", 0.01)),
        batch_size=int(model_config.get("batch_size", 32)),
        max_epochs=int(model_config.get("max_epochs", 50)),
        patience=int(model_config.get("patience", 8)),
        denoising_rate=float(model_config.get("denoising_rate", 0.15)),
        gradient_clip=float(model_config.get("gradient_clip", 5.0)),
        seed=seed,
    )


def _write_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    frame: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    spec: DatasetSpec,
    preprocessor: MixedTypePreprocessor,
    model: NumpyVAE,
    structured_model: NumpyVAE | None,
    test_encoded,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
    efficiency: pd.DataFrame,
    masking_metadata: list[dict],
    explanation_records: list[dict],
    skipped_methods: dict[str, str],
    elapsed_seconds: float,
) -> None:
    resolved_config = {
        key: value for key, value in config.items() if not str(key).startswith("_")
    }
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)

    metrics.to_csv(output_dir / "metrics_long.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    statistics.to_csv(output_dir / "statistical_tests.csv", index=False)
    efficiency.to_csv(output_dir / "efficiency.csv", index=False)
    pd.DataFrame(model.history_as_dicts()).to_csv(
        output_dir / "training_history.csv", index=False
    )
    if structured_model is not None:
        pd.DataFrame(structured_model.history_as_dicts()).to_csv(
            output_dir / "training_history_structured.csv", index=False
        )

    cohort = {
        "total_records": int(len(frame)),
        "total_patients": int(frame[spec.patient_id].nunique()),
        "partitions": {
            "train": _partition_summary(train, spec),
            "validation": _partition_summary(validation, spec),
            "test": _partition_summary(test, spec),
        },
        "natural_missing_fraction": {
            column: float(frame[column].isna().mean()) for column in spec.features
        },
        "targets": list(spec.targets),
    }
    _write_json(output_dir / "cohort_summary.json", cohort)
    _write_json(output_dir / "masking_metadata.json", masking_metadata)
    _write_json(output_dir / "neighbor_explanations.json", explanation_records)
    _write_json(output_dir / "preprocessor_metadata.json", preprocessor.field_metadata())
    _write_json(
        output_dir / "run_metadata.json",
        {
            "status": "completed",
            "elapsed_seconds": float(elapsed_seconds),
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "vae_parameter_count": model.parameter_count(),
            "vae_best_epoch": model.best_epoch,
            "metric_rows": int(len(metrics)),
            "methods": sorted(metrics["method"].unique().tolist()),
            "mechanisms": sorted(metrics["mechanism"].unique().tolist()),
            "mask_rates": sorted(
                float(value) for value in metrics["mask_rate"].unique().tolist()
            ),
            "masking_seeds": sorted(
                int(value) for value in metrics["mask_seed"].unique().tolist()
            ),
            "implementation": {
                "vae_backend": "NumPy reference implementation",
                "clinical_result_reproduction": False,
            },
            "structured_vae_best_epoch": (
                structured_model.best_epoch if structured_model is not None else None
            ),
            "skipped_methods": skipped_methods,
            "result_scope": (
                "Output from the resolved configuration. Synthetic runs validate software "
                "behavior only. A clinical run must not be labeled a reproduction of the "
                "manuscript without matching data provenance, masks, checkpoints, and "
                "per-seed prediction artifacts."
            ),
        },
    )
    _plot_summary(summary, output_dir / "metrics_summary.png")
    _plot_latent_space(
        latent=model.encode(test_encoded),
        test=test,
        spec=spec,
        config=config,
        path=output_dir / "latent_space_pca.png",
    )
    _plot_efficiency(efficiency, output_dir / "efficiency_profile.png")


def _partition_summary(frame: pd.DataFrame, spec: DatasetSpec) -> dict:
    return {
        "records": int(len(frame)),
        "patients": int(frame[spec.patient_id].nunique()),
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _peak_rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return usage / (1024**2)
    return usage / 1024


def _plot_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = list(summary["metric"].drop_duplicates())
    figure, axes = plt.subplots(
        nrows=len(metric_names),
        ncols=1,
        figsize=(9, max(3.2, 2.8 * len(metric_names))),
        squeeze=False,
    )
    for axis, metric in zip(axes[:, 0], metric_names):
        selected = summary[summary["metric"] == metric].sort_values("mean")
        axis.barh(selected["method"], selected["mean"], color="#36459B")
        axis.set_title(f"{metric} (synthetic smoke run unless configured otherwise)")
        axis.set_xlabel(metric)
        axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_latent_space(
    latent: np.ndarray,
    test: pd.DataFrame,
    spec: DatasetSpec,
    config: dict[str, Any],
    path: Path,
) -> None:
    """Write a diagnostic two-dimensional PCA view of posterior means."""

    if len(latent) == 0:
        return
    if latent.shape[1] >= 2:
        coordinates = PCA(n_components=2).fit_transform(latent)
    else:
        coordinates = np.column_stack([latent[:, 0], np.zeros(len(latent))])

    requested_color = config.get("visualization", {}).get("color_by")
    fallback_color = next(
        (target for target in spec.targets if target in spec.categorical),
        spec.categorical[0] if spec.categorical else None,
    )
    color_by = requested_color or fallback_color
    if color_by is not None and color_by not in test.columns:
        raise ValueError(f"visualization.color_by column not found: {color_by}")

    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    if color_by is None:
        axis.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.75, s=30)
    else:
        labels = test[color_by].fillna("__MISSING__").astype(str).to_numpy()
        for label in sorted(set(labels)):
            selected = labels == label
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                alpha=0.78,
                s=32,
                label=label,
            )
        axis.legend(title=color_by, fontsize=8, loc="best")
    axis.set_xlabel("Latent PCA component 1")
    axis.set_ylabel("Latent PCA component 2")
    axis.set_title("Diagnostic PCA projection of VAE posterior means")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_efficiency(efficiency: pd.DataFrame, path: Path) -> None:
    """Write a compact training/inference efficiency diagnostic."""

    if efficiency.empty:
        return
    selected = efficiency.sort_values(
        "mean_inference_seconds_per_test_partition", ascending=True
    )
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.2))
    axes[0].barh(
        selected["method"],
        selected["training_seconds"].fillna(0.0),
        color="#5767B2",
    )
    axes[0].set_xlabel("Training seconds")
    axes[0].set_title("Training time")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(
        selected["method"],
        selected["mean_inference_seconds_per_test_partition"].fillna(0.0),
        color="#C46D3B",
    )
    axes[1].set_xlabel("Mean seconds per test partition")
    axes[1].set_title("Inference time")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle("Measured efficiency profile for this run")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
