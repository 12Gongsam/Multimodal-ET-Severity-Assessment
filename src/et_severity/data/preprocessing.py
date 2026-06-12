"""Manifest construction and sensor-file indexing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from ..config import DIR_RE, TASK2IDX


def build_manifest(
    label_csv_path,
    root_dir,
    *,
    target_col: str = "target",
    require_exists: bool = True,
) -> pd.DataFrame:
    """Build the file-level training manifest from the label CSV."""
    label_csv_path = Path(label_csv_path)
    root_dir = Path(root_dir)
    labels = pd.read_csv(label_csv_path)

    required = {"file", "patient_id", "session", "task", target_col}
    missing = required - set(labels.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for row in tqdm(
        labels.itertuples(index=False),
        total=len(labels),
        desc="Build manifest",
        unit="row",
    ):
        raw_file = str(getattr(row, "file")).strip()
        if not raw_file or raw_file.lower() == "nan":
            continue

        path = Path(raw_file)
        if not path.is_absolute():
            path = (root_dir / path).resolve()
        if require_exists and not path.exists():
            continue

        target_value = getattr(row, target_col)
        try:
            target = int(target_value)
        except (TypeError, ValueError):
            target = int(float(target_value))

        patient_match = re.search(r"(\d+)", str(getattr(row, "patient_id")))
        patient_id = int(patient_match.group(1)) if patient_match else -1

        session_text = str(getattr(row, "session"))
        session_match = DIR_RE.search(session_text) or DIR_RE.search(str(path))
        if session_match:
            session = int(session_match.group(2))
            date = session_match.group(3)
        else:
            session = -1
            date = "00000000"

        task = str(getattr(row, "task"))
        rows.append(
            {
                "path": path,
                "patient_id": patient_id,
                "session": session,
                "date": date,
                "task": task,
                "task_idx": TASK2IDX.get(task.lower(), TASK2IDX["unknown"]),
                "target": target,
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest

    minimum_target = manifest["target"].min()
    if minimum_target > 0:
        manifest["target"] -= minimum_target

    manifest = manifest.sort_values(
        ["patient_id", "session", "task", "path"]
    ).reset_index(drop=True)
    manifest["target"] = manifest["target"].astype(int)
    return manifest


def split_train_valid_by_patient(
    manifest: pd.DataFrame,
    val_pid: Sequence[int],
    *,
    pid_col: str = "patient_id",
    reset_index: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a manifest into training and validation patients."""
    if pid_col not in manifest.columns:
        raise KeyError(
            f"Missing patient column {pid_col!r}. "
            f"Available columns: {list(manifest.columns)}"
        )

    patient_ids = pd.to_numeric(manifest[pid_col], errors="coerce")
    validation_ids = {int(value) for value in val_pid}
    validation_mask = patient_ids.isin(validation_ids)

    train_manifest = manifest.loc[~validation_mask].copy()
    valid_manifest = manifest.loc[validation_mask].copy()
    if reset_index:
        train_manifest.reset_index(drop=True, inplace=True)
        valid_manifest.reset_index(drop=True, inplace=True)
    return train_manifest, valid_manifest


def build_segment_manifest(
    manifest: pd.DataFrame,
    *,
    seg_len: int,
    hop: int,
    usecols: Sequence[str],
    require_exists: bool = True,
    keep_tail: bool = False,
) -> pd.DataFrame:
    """Expand each sensor CSV into fixed-length segment index rows."""
    required = {
        "path",
        "patient_id",
        "session",
        "date",
        "task",
        "task_idx",
        "target",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    segment_rows = []
    segment_hop = seg_len if hop is None or hop <= 0 else int(hop)

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Expand to segments",
        unit="file",
    ):
        path = Path(str(row.path))
        if require_exists and not path.exists():
            continue

        try:
            header = pd.read_csv(path, nrows=0)
            available = [column for column in usecols if column in header.columns]
            row_count = len(
                pd.read_csv(path, usecols=[available[0]])
                if available
                else pd.read_csv(path)
            )
        except (OSError, ValueError, pd.errors.ParserError):
            continue

        if row_count <= 0 or (row_count < seg_len and not keep_tail):
            continue

        starts = list(
            range(0, max(0, row_count - seg_len + 1), segment_hop)
        )
        if row_count < seg_len:
            starts = [0]
        elif keep_tail:
            tail_start = row_count - seg_len
            if not starts or starts[-1] != tail_start:
                starts.append(tail_start)

        for segment_index, start in enumerate(starts):
            segment_rows.append(
                {
                    "path": str(path),
                    "patient_id": int(row.patient_id),
                    "session": int(row.session),
                    "date": str(row.date),
                    "task": str(row.task),
                    "task_idx": int(row.task_idx),
                    "target": int(row.target),
                    "start": int(start),
                    "end": int(min(start + seg_len, row_count)),
                    "seg_idx": segment_index,
                }
            )

    segments = pd.DataFrame(segment_rows)
    if segments.empty:
        return segments
    return segments.sort_values(
        ["patient_id", "session", "task", "path", "start"]
    ).reset_index(drop=True)
