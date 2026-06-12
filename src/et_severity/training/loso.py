"""Leave-one-subject-out training and metric aggregation."""

from __future__ import annotations

import gc
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..models.joint_instance_attention_mil import JointInstanceAttentionConfig
from ..models.mil_models import MultimodalModelConfig, SingleModelConfig
from .engine import TrainingConfig
from .experiments import fit_multimodal, fit_single_modality

METRIC_COLUMNS = ("accuracy", "precision_w", "f1_w")


def calculate_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate the project-standard patient-level classification metrics."""
    precision, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_w": float(precision),
        "f1_w": float(f1),
    }


def run_loso_cv(
    loaders,
    *,
    modality: str,
    model_config,
    training_config: TrainingConfig,
    target: str = "severity",
    seed: int = 42,
    expected_folds: Optional[int] = 9,
    checkpoint_dir=None,
):
    """Train all LOSO folds and return patient-level and aggregate metrics."""
    if modality not in {"acc", "traj", "multimodal"}:
        raise ValueError("modality must be 'acc', 'traj', or 'multimodal'")
    if target not in {"severity", "task"}:
        raise ValueError("target must be 'severity' or 'task'")
    if modality == "multimodal" and target != "severity":
        raise ValueError("multimodal training currently supports severity only")
    if modality == "multimodal" and not isinstance(
        model_config,
        (MultimodalModelConfig, JointInstanceAttentionConfig),
    ):
        raise TypeError(
            "multimodal training requires MultimodalModelConfig or "
            "JointInstanceAttentionConfig"
        )
    if modality != "multimodal" and not isinstance(
        model_config, SingleModelConfig
    ):
        raise TypeError("single-modality training requires SingleModelConfig")
    if expected_folds is not None and len(loaders) != expected_folds:
        raise ValueError(
            f"Expected {expected_folds} LOSO folds, but received {len(loaders)}."
        )

    output_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows = []
    prediction_frames = []
    histories = {}
    fold_ids = sorted(loaders, key=lambda value: int(value))

    for fold_index, patient_id in enumerate(fold_ids, start=1):
        print(
            f"\n{'=' * 60}\n"
            f"Fold {fold_index}/{len(fold_ids)} | "
            f"Validation patient: {patient_id}\n"
            f"{'=' * 60}"
        )
        train_loader, valid_loader = loaders[patient_id]

        if modality == "multimodal":
            run = fit_multimodal(
                train_loader,
                valid_loader,
                model_config=model_config,
                training_config=training_config,
                seed=seed,
            )
        else:
            run = fit_single_modality(
                train_loader,
                valid_loader,
                modality=modality,
                target=target,
                model_config=model_config,
                training_config=training_config,
                seed=seed,
            )

        y_true = run.metrics["y_true"].cpu().numpy()
        y_pred = run.metrics["y_pred"].cpu().numpy()
        metrics = calculate_classification_metrics(y_true, y_pred)
        checkpoint_path = None

        if output_dir is not None:
            if modality == "multimodal":
                architecture = (
                    "JointInstanceAttention_"
                    if isinstance(model_config, JointInstanceAttentionConfig)
                    else ""
                )
                checkpoint_name = (
                    f"{architecture}ACC_{model_config.acc_encoder.name}_"
                    f"Traj_{model_config.traj_encoder.name}_"
                    f"{target}_val_pid{patient_id}.pth"
                )
            else:
                checkpoint_name = (
                    f"{model_config.encoder.name}_{modality}_{target}_"
                    f"val_pid{patient_id}.pth"
                )
            checkpoint_path = output_dir / checkpoint_name
            torch.save(
                {
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in run.model.state_dict().items()
                    },
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "patient_id": int(patient_id),
                    "metrics": metrics,
                },
                checkpoint_path,
            )

        fold_rows.append(
            {
                "fold": fold_index,
                "patient_id": int(patient_id),
                "n": len(y_true),
                **metrics,
                "checkpoint_path": (
                    None if checkpoint_path is None else str(checkpoint_path)
                ),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_index,
                    "patient_id": int(patient_id),
                    f"true_{target}": y_true,
                    f"pred_{target}": y_pred,
                }
            )
        )
        histories[str(patient_id)] = run.history
        print(
            f"accuracy={metrics['accuracy']:.4f} | "
            f"precision_w={metrics['precision_w']:.4f} | "
            f"f1_w={metrics['f1_w']:.4f}"
        )

        del run
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_summary = (
        fold_results.loc[:, list(METRIC_COLUMNS)]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    return {
        "fold_results": fold_results,
        "fold_summary": fold_summary,
        "predictions": predictions,
        "histories": histories,
    }
