"""Prediction-summary helpers used by the analysis notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from .metrics import compute_patient_level_metrics, summarize_metric_columns

DEFAULT_MODEL_NAMES = ("LSTM", "ResNet18", "TimesNet", "MyWaveNet")


def summarize_prediction_file(
    csv_path,
    *,
    true_col: str = "true_severity",
    pred_col: str = "pred_severity",
    metric_columns: Sequence[str] = ("accuracy", "precision_w", "f1_w"),
) -> Mapping[str, object]:
    """Load one prediction CSV and return patient-level and aggregate summaries."""
    csv_path = Path(csv_path)
    prediction_df = pd.read_csv(csv_path)
    patient_metrics = compute_patient_level_metrics(
        prediction_df,
        true_col=true_col,
        pred_col=pred_col,
        auto_detect=False,
    )
    return {
        "csv_path": str(csv_path),
        "predictions": prediction_df,
        "patient_metrics": patient_metrics,
        "summary": summarize_metric_columns(patient_metrics, metric_columns=metric_columns),
    }


def _summary_row_from_metrics(
    label: str,
    csv_path: Path,
    patient_metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
) -> dict:
    summary = summarize_metric_columns(patient_metrics, metric_columns=metric_columns).set_index("metric")
    row = {"label": label, "csv_path": str(csv_path)}
    for metric_name in metric_columns:
        row[f"{metric_name}_mean"] = float(summary.loc[metric_name, "mean"])
        row[f"{metric_name}_std"] = float(summary.loc[metric_name, "std"])
    return row


def summarize_prediction_collection(
    label_to_path: Mapping[str, Path],
    *,
    true_col: str = "true_severity",
    pred_col: str = "pred_severity",
    metric_columns: Sequence[str] = ("accuracy", "precision_w", "f1_w"),
) -> pd.DataFrame:
    """Summarize a labeled set of prediction CSV files."""
    rows = []
    for label, csv_path in label_to_path.items():
        result = summarize_prediction_file(
            csv_path,
            true_col=true_col,
            pred_col=pred_col,
            metric_columns=metric_columns,
        )
        rows.append(
            _summary_row_from_metrics(
                label,
                Path(csv_path),
                result["patient_metrics"],
                metric_columns=metric_columns,
            )
        )
    return pd.DataFrame(rows)


def summarize_multimodal_prediction_runs(
    predictions_dir,
    *,
    acc_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    traj_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    time_pool: Optional[str] = None,
) -> pd.DataFrame:
    """Summarize all multimodal severity prediction CSVs into one dataframe."""
    predictions_dir = Path(predictions_dir)
    rows = []
    for acc_name in acc_names:
        for traj_name in traj_names:
            filename = f"ACC_{acc_name}_Traj_{traj_name}_val_predictions.csv"
            if time_pool:
                filename = f"{time_pool}_{filename}"
            csv_path = predictions_dir / "multi" / filename
            result = summarize_prediction_file(csv_path)
            row = _summary_row_from_metrics(
                f"{acc_name} + {traj_name}",
                csv_path,
                result["patient_metrics"],
                metric_columns=("accuracy", "precision_w", "f1_w"),
            )
            row.update(
                {
                    "acc_encoder": acc_name,
                    "traj_encoder": traj_name,
                    "time_pool": time_pool or "attn",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_single_modality_prediction_runs(
    predictions_dir,
    *,
    modality: str,
    target: str,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> pd.DataFrame:
    """Summarize single-modality prediction CSVs for one target/modality pair."""
    if target not in {"severity", "task"}:
        raise ValueError("target must be 'severity' or 'task'")

    predictions_dir = Path(predictions_dir)
    true_col = "true_severity" if target == "severity" else "true_task"
    pred_col = "pred_severity" if target == "severity" else "pred_task"

    rows = []
    for model_name in model_names:
        csv_path = predictions_dir / "single" / target / f"{model_name}_{modality}_{target}_val_predictions.csv"
        result = summarize_prediction_file(csv_path, true_col=true_col, pred_col=pred_col)
        row = _summary_row_from_metrics(
            model_name,
            csv_path,
            result["patient_metrics"],
            metric_columns=("accuracy", "precision_w", "f1_w"),
        )
        row.update(
            {
                "model_name": model_name,
                "modality": modality,
                "target": target,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_single_vs_multimodal_accuracy_report(
    predictions_dir,
    *,
    modality: str = "acc",
    target: str = "severity",
    primary_encoder_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    secondary_encoder_names: Sequence[str] = DEFAULT_MODEL_NAMES,
) -> pd.DataFrame:
    """Build the grouped-bar report comparing single vs. multimodal accuracy."""
    predictions_dir = Path(predictions_dir)
    rows = []

    for primary_name in primary_encoder_names:
        single_csv = (
            predictions_dir / "single" / target / f"{primary_name}_{modality}_{target}_val_predictions.csv"
        )
        single_metrics = summarize_prediction_file(single_csv)["patient_metrics"]
        patient_order = single_metrics["patient_id"].sort_values().tolist()

        single_accuracy = (
            single_metrics
            .set_index("patient_id")
            .loc[patient_order, "accuracy"]
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "encoder_name": primary_name,
                "run_label": "Single",
                "run_group": 0,
                "mean_accuracy_pct": float(single_accuracy.mean() * 100.0),
                "std_accuracy_pct": float(single_accuracy.std(ddof=1) * 100.0) if len(single_accuracy) > 1 else 0.0,
                "p_value_vs_single": np.nan,
                "csv_path": str(single_csv),
            }
        )

        for run_group, secondary_name in enumerate(secondary_encoder_names, start=1):
            multi_csv = (
                predictions_dir / "multi" / f"ACC_{primary_name}_Traj_{secondary_name}_val_predictions.csv"
            )
            multi_metrics = summarize_prediction_file(multi_csv)["patient_metrics"]
            multi_accuracy = (
                multi_metrics
                .set_index("patient_id")
                .loc[patient_order, "accuracy"]
                .to_numpy(dtype=float)
            )
            _, p_value = ttest_rel(single_accuracy, multi_accuracy)

            rows.append(
                {
                    "encoder_name": primary_name,
                    "run_label": f"Multi Traj({secondary_name})",
                    "run_group": run_group,
                    "mean_accuracy_pct": float(multi_accuracy.mean() * 100.0),
                    "std_accuracy_pct": float(multi_accuracy.std(ddof=1) * 100.0) if len(multi_accuracy) > 1 else 0.0,
                    "p_value_vs_single": float(p_value),
                    "csv_path": str(multi_csv),
                }
            )

    return pd.DataFrame(rows)
