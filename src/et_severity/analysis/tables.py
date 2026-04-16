"""Table-building helpers for the analysis notebook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_session_severity_table(
    df: pd.DataFrame,
    *,
    severity_col: str = "target_k5",
    patient_col: str = "patient_id",
    session_col: str = "session",
    tetras_col: str = "tetras_score",
) -> pd.DataFrame:
    """Recreate the paper table summarizing per-session severity counts."""
    working = df.copy()
    severity_levels = [1, 2, 3, 4]
    severity_values = working[severity_col].astype(int) + 1

    counts = (
        pd.crosstab(
            [working[patient_col], working[session_col]],
            severity_values,
        )
        .reindex(columns=severity_levels, fill_value=0)
    )
    counts.columns = [f"Sev {level}" for level in counts.columns]
    counts["Total"] = counts.sum(axis=1)

    weights = pd.Series({"Sev 1": 1, "Sev 2": 2, "Sev 3": 3, "Sev 4": 4})
    mean_severity = (
        counts[["Sev 1", "Sev 2", "Sev 3", "Sev 4"]]
        .mul(weights, axis=1)
        .sum(axis=1)
        / counts["Total"]
    ).round(2)
    counts.insert(counts.columns.get_loc("Total"), "Mean Sev", mean_severity)

    tetras = (
        working.groupby([patient_col, session_col])[tetras_col]
        .first()
        .reindex(counts.index)
    )
    counts.insert(counts.columns.get_loc("Total"), "TETRAS", tetras)

    return (
        counts
        .reset_index()
        .rename(columns={patient_col: "Patient", session_col: "Session"})
    )


def build_session_severity_table_from_csv(csv_path, **kwargs) -> pd.DataFrame:
    """Load the label CSV and build the paper table."""
    return build_session_severity_table(pd.read_csv(csv_path), **kwargs)


def save_table_csv(table_df: pd.DataFrame, output_path) -> Path:
    """Save a generated table and return the resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False)
    return output_path
