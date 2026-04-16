"""Clean analysis utilities for patient-level metrics and ordinal agreement."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def _coerce_label_series(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        return series.astype(int)

    as_numeric = pd.to_numeric(series, errors="coerce")
    if as_numeric.notna().all():
        return as_numeric.astype(int)
    return series.astype(str)


def compute_patient_level_metrics(
    df: pd.DataFrame,
    *,
    id_col: str = "patient_id",
    true_col: str = "true_severity",
    pred_col: str = "pred_severity",
    drop_na: bool = True,
    auto_detect: bool = True,
) -> pd.DataFrame:
    """Compute patient-level accuracy, weighted precision, recall, and F1."""
    data = df.copy()

    if auto_detect:
        required = {id_col, true_col, pred_col}
        if not required.issubset(data.columns):
            if {"true_task", "pred_task"}.issubset(data.columns):
                true_col, pred_col = "true_task", "pred_task"
            else:
                missing = sorted(required - set(data.columns))
                raise KeyError(f"Missing required columns: {missing}")
    else:
        missing = sorted({id_col, true_col, pred_col} - set(data.columns))
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    if drop_na:
        data = data.dropna(subset=[id_col, true_col, pred_col])

    data[true_col] = _coerce_label_series(data[true_col])
    data[pred_col] = _coerce_label_series(data[pred_col])

    rows = []
    for patient_id, group in data.groupby(id_col, sort=True):
        y_true = group[true_col].values
        y_pred = group[pred_col].values
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        rows.append(
            {
                id_col: patient_id,
                "n": len(group),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_w": precision,
                "recall_w": recall,
                "f1_w": f1,
            }
        )

    return pd.DataFrame(rows)


def summarize_metric_columns(
    metrics_df: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = ("accuracy", "precision_w", "f1_w"),
) -> pd.DataFrame:
    """Return mean and sample-standard-deviation for selected metric columns."""
    numeric = metrics_df.loc[:, list(metric_columns)].apply(pd.to_numeric, errors="coerce")
    rows = []
    for column in metric_columns:
        values = numeric[column].dropna()
        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()) if not values.empty else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "count": int(values.count()),
            }
        )
    return pd.DataFrame(rows)


def compare_ordinal_ratings(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    *,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Blues",
) -> Mapping[str, object]:
    """Compare two ordinal label columns and optionally draw a confusion matrix."""
    clean = df[[true_col, pred_col]].dropna()
    y_true = clean[true_col]
    y_pred = clean[pred_col]

    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
    quadratic_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("Reference label", fontsize=14)

    return {
        "n": int(len(clean)),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "quadratic_weighted_kappa": float(quadratic_kappa),
        "labels": labels,
        "confusion_matrix": matrix,
        "ax": ax,
    }


def median_absolute_deviation(values: Iterable[float]) -> float:
    """Return the median absolute deviation around the sample median."""
    array = np.asarray(list(values), dtype=float)
    median = np.median(array)
    return float(np.median(np.abs(array - median)))
