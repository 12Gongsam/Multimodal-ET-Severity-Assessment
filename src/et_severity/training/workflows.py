"""Data preparation and LOSO experiment workflows."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DEFAULT_TARGET_PER_CLASS
from ..data.datasets import build_LOSO_loaders
from ..data.preprocessing import build_manifest, split_and_delete_multidirect_fs
from ..models.mil_models import (
    MultimodalModelConfig,
    SingleModelConfig,
    build_multimodal_model,
    build_single_modality_model,
)
from ..utils.reproducibility import set_seed
from .engine import (
    TrainingConfig,
    evaluate_multimodal_model,
    evaluate_single_model,
    train_multimodal_model,
    train_single_model,
)


def prepare_training_data(
    root_dir,
    label_csv_path,
    *,
    target_col: str = "target_k5",
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    filter_task: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    modality: str = "multimodal",
    patient_ids: Optional[Sequence[int]] = None,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    split_multidirect: bool = False,
):
    """Build a manifest and modality-specific LOSO loaders."""
    root_dir = Path(root_dir)
    label_csv_path = Path(label_csv_path)

    split_result = None
    if split_multidirect:
        split_result = split_and_delete_multidirect_fs(root_dir)

    manifest = build_manifest(label_csv_path, root_dir, target_col=target_col)
    loaders = build_LOSO_loaders(
        manifest,
        target_per_class=target_per_class,
        filter_task=filter_task,
        batch_size=batch_size,
        modality=modality,
        patient_ids=patient_ids,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    return manifest, loaders, split_result


def summarize_fold_results(results_df: pd.DataFrame) -> Dict[str, float]:
    if results_df.empty:
        return {
            "acc_mean": 0.0,
            "acc_std": 0.0,
            "macro_f1_mean": 0.0,
            "macro_f1_std": 0.0,
        }

    return {
        "acc_mean": float(results_df["acc"].mean()),
        "acc_std": (
            float(results_df["acc"].std(ddof=1))
            if len(results_df) > 1
            else 0.0
        ),
        "macro_f1_mean": float(results_df["macro_f1"].mean()),
        "macro_f1_std": (
            float(results_df["macro_f1"].std(ddof=1))
            if len(results_df) > 1
            else 0.0
        ),
    }


def _save_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir,
    filename: str,
    *,
    model_config,
    training_config: TrainingConfig,
    fold_metrics: Dict[str, object],
) -> Optional[Path]:
    if checkpoint_dir is None:
        return None

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "fold_metrics": {
                "acc": float(fold_metrics["acc"]),
                "macro_f1": float(fold_metrics["macro_f1"]),
                "per_class_acc": fold_metrics["per_class_acc"],
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def run_multimodal_severity_loso(
    loaders,
    *,
    model_config: MultimodalModelConfig,
    training_config: TrainingConfig = TrainingConfig(),
    device: Optional[torch.device] = None,
    checkpoint_dir=None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, list]]]:
    """Train one configurable multimodal severity model across LOSO folds."""
    device = DEFAULT_DEVICE if device is None else device
    rows = []
    histories = {}

    for patient_id, (train_loader, valid_loader) in loaders.items():
        print(f"[Fold PID={patient_id}]")
        set_seed(seed)
        model = build_multimodal_model(model_config).to(device)
        history = train_multimodal_model(
            model,
            train_loader,
            valid_loader,
            device=device,
            config=training_config,
        )
        metrics = evaluate_multimodal_model(
            model,
            valid_loader,
            device=device,
        )
        filename = (
            f"ACC_{model_config.acc_encoder.name}_"
            f"Traj_{model_config.traj_encoder.name}_"
            f"val_pid{patient_id}.pth"
        )
        checkpoint_path = _save_checkpoint(
            model,
            checkpoint_dir,
            filename,
            model_config=model_config,
            training_config=training_config,
            fold_metrics=metrics,
        )

        print(
            f"[Fold PID={patient_id}] severity ACC={metrics['acc']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"per-class acc={np.round(metrics['per_class_acc'], 3).tolist()}"
        )
        rows.append(
            {
                "patient_id": str(patient_id),
                "acc": float(metrics["acc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "checkpoint_path": (
                    None if checkpoint_path is None else str(checkpoint_path)
                ),
            }
        )
        histories[str(patient_id)] = history

    results_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return results_df, summarize_fold_results(results_df), histories


def run_single_modality_loso(
    loaders,
    *,
    model_config: SingleModelConfig,
    modality: str,
    target: str = "severity",
    training_config: TrainingConfig = TrainingConfig(),
    device: Optional[torch.device] = None,
    checkpoint_dir=None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, list]]]:
    """Train one configurable single-modality model across LOSO folds."""
    if modality not in {"acc", "traj"}:
        raise ValueError("modality must be 'acc' or 'traj'")
    if target not in {"severity", "task"}:
        raise ValueError("target must be 'severity' or 'task'")

    device = DEFAULT_DEVICE if device is None else device
    rows = []
    histories = {}

    for patient_id, (train_loader, valid_loader) in loaders.items():
        print(f"[Fold PID={patient_id}]")
        set_seed(seed)
        model = build_single_modality_model(
            model_config,
            modality=modality,
        ).to(device)
        history = train_single_model(
            model,
            train_loader,
            valid_loader,
            device=device,
            modality=modality,
            target=target,
            config=training_config,
        )
        metrics = evaluate_single_model(
            model,
            valid_loader,
            device=device,
            modality=modality,
            target=target,
        )
        filename = (
            f"{model_config.encoder.name}_{modality}_{target}_"
            f"val_pid{patient_id}.pth"
        )
        checkpoint_path = _save_checkpoint(
            model,
            checkpoint_dir,
            filename,
            model_config=model_config,
            training_config=training_config,
            fold_metrics=metrics,
        )

        print(
            f"[Fold PID={patient_id}] {target} ACC={metrics['acc']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f} | "
            f"per-class acc={np.round(metrics['per_class_acc'], 3).tolist()}"
        )
        rows.append(
            {
                "patient_id": str(patient_id),
                "acc": float(metrics["acc"]),
                "macro_f1": float(metrics["macro_f1"]),
                "checkpoint_path": (
                    None if checkpoint_path is None else str(checkpoint_path)
                ),
            }
        )
        histories[str(patient_id)] = history

    results_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return results_df, summarize_fold_results(results_df), histories
