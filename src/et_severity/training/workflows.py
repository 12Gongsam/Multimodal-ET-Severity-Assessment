"""High-level training workflows used by the import-based notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_D_MODEL,
    DEFAULT_MIL_ATTN_DIM,
    DEFAULT_TARGET_PER_CLASS,
)
from ..data.datasets import build_LOSO_loaders
from ..data.preprocessing import build_manifest, split_and_delete_multidirect_fs
from ..models.mil_models import MIL_MultiModal, MIL_Single, build_model
from ..utils.reproducibility import set_seed
from .engine import (
    evaluate_on_loader_sev_only,
    evaluate_on_loader_single,
    train_model_sev_only,
    train_model_single,
)


def prepare_training_data(
    root_dir,
    label_csv_path,
    *,
    target_col: str = "target_k5",
    target_per_class: int = DEFAULT_TARGET_PER_CLASS,
    filter_task: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    split_multidirect: bool = True,
):
    """Build the manifest and LOSO loaders used by Main.ipynb."""
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
    )
    return manifest, loaders, split_result


def summarize_fold_results(results_df: pd.DataFrame) -> Dict[str, float]:
    """Return mean/std summary statistics for a LOSO result table."""
    if results_df.empty:
        return {"acc_mean": 0.0, "acc_std": 0.0, "macro_f1_mean": 0.0, "macro_f1_std": 0.0}

    acc_std = float(results_df["acc"].std(ddof=1)) if len(results_df) > 1 else 0.0
    f1_std = float(results_df["macro_f1"].std(ddof=1)) if len(results_df) > 1 else 0.0
    return {
        "acc_mean": float(results_df["acc"].mean()),
        "acc_std": acc_std,
        "macro_f1_mean": float(results_df["macro_f1"].mean()),
        "macro_f1_std": f1_std,
    }


def _save_checkpoint(model: torch.nn.Module, checkpoint_dir, filename: str) -> Optional[Path]:
    if checkpoint_dir is None:
        return None

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def run_multimodal_severity_loso(
    loaders,
    *,
    acc_name: str,
    traj_name: str,
    device: Optional[torch.device] = None,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    patience: int = 20,
    d_model: int = DEFAULT_D_MODEL,
    mil_attn_dim: int = DEFAULT_MIL_ATTN_DIM,
    time_pool: str = "attn",
    ma_heads: int = 8,
    checkpoint_dir=None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, list]]]:
    """Train the multimodal severity model across LOSO folds."""
    device = DEFAULT_DEVICE if device is None else device

    rows = []
    histories = {}

    for patient_id, (dl_tr, dl_va) in loaders.items():
        print(f"[Fold PID={patient_id}]")
        set_seed(seed)

        acc_encoder = build_model(model_name=acc_name, in_ch=3, num_classes=4)
        traj_encoder = build_model(model_name=traj_name, in_ch=2, num_classes=3)

        model = MIL_MultiModal(
            acc_encoder=acc_encoder,
            traj_encoder=traj_encoder,
            d_model=d_model,
            num_sclass=4,
            ma_heads=ma_heads,
            mil_attn_dim=mil_attn_dim,
            time_pool=time_pool,
        ).to(device)

        history = train_model_sev_only(
            model,
            train_loader=dl_tr,
            valid_loader=dl_va,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=patience,
            weight_decay=1e-4,
            grad_clip=1.0,
            sched_factor=0.5,
            sched_patience=5,
        )
        metrics = evaluate_on_loader_sev_only(model, dl_va, device)
        checkpoint_path = _save_checkpoint(
            model,
            checkpoint_dir,
            f"ACC_{acc_name}_Traj_{traj_name}_val_pid{patient_id}.pth",
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
                "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            }
        )
        histories[str(patient_id)] = history

    results_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return results_df, summarize_fold_results(results_df), histories


def run_single_modality_loso(
    loaders,
    *,
    model_name: str,
    modality: str,
    target: str = "severity",
    device: Optional[torch.device] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 20,
    d_model: int = DEFAULT_D_MODEL,
    mil_attn_dim: int = DEFAULT_MIL_ATTN_DIM,
    checkpoint_dir=None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, list]]]:
    """Train a single-modality MIL model across LOSO folds."""
    if modality not in {"acc", "traj"}:
        raise ValueError("modality must be 'acc' or 'traj'")
    if target not in {"severity", "task"}:
        raise ValueError("target must be 'severity' or 'task'")

    device = DEFAULT_DEVICE if device is None else device
    in_ch = 3 if modality == "acc" else 2
    num_classes = 4 if target == "severity" else 3

    rows = []
    histories = {}

    for patient_id, (dl_tr, dl_va) in loaders.items():
        print(f"[Fold PID={patient_id}]")
        set_seed(seed)

        base_encoder = build_model(
            model_name=model_name,
            in_ch=in_ch,
            num_classes=num_classes,
        )
        model = MIL_Single(
            base_encoder=base_encoder,
            num_classes=num_classes,
            d_model=d_model,
            time_dropout=0.1,
            attn_dim=mil_attn_dim,
            mil_dropout=0.1,
        ).to(device)

        history = train_model_single(
            model,
            train_loader=dl_tr,
            valid_loader=dl_va,
            device=device,
            modality=modality,
            target=target,
            epochs=epochs,
            lr=learning_rate,
            early_stopping_patience=patience,
            weight_decay=1e-4,
            grad_clip=1.0,
            sched_factor=0.5,
            sched_patience=5,
        )
        metrics = evaluate_on_loader_single(
            model,
            loader=dl_va,
            device=device,
            target=target,
            modality=modality,
        )
        checkpoint_path = _save_checkpoint(
            model,
            checkpoint_dir,
            f"{model_name}_{modality}_{target}_val_pid{patient_id}.pth",
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
                "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            }
        )
        histories[str(patient_id)] = history

    results_df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    return results_df, summarize_fold_results(results_df), histories


prepare_loso_training_data = prepare_training_data
run_multimodal_loso_severity_experiment = run_multimodal_severity_loso
run_single_modality_loso_experiment = run_single_modality_loso
