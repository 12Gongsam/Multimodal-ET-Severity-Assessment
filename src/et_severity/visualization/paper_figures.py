"""High-level paper figures extracted from the analysis notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr

from ..analysis.metrics import median_absolute_deviation
from .statistical_plots import add_significance_bracket


def _save_if_requested(fig: plt.Figure, save_path) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


def plot_single_vs_multimodal_accuracy(
    report_df: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    y_limit: tuple[float, float] = (0.0, 150.0),
) -> plt.Axes:
    """Plot the grouped-bar comparison between single and multimodal accuracy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ordered_runs = ["Single", "Multi Traj(LSTM)", "Multi Traj(ResNet18)", "Multi Traj(TimesNet)", "Multi Traj(MyWaveNet)"]
    display_labels = {
        "Single": "Single",
        "Multi Traj(LSTM)": "Multi Traj(LSTM)",
        "Multi Traj(ResNet18)": "Multi Traj(ResNet18)",
        "Multi Traj(TimesNet)": "Multi Traj(TimesNet)",
        "Multi Traj(MyWaveNet)": "Multi Traj(WaveNet-style)",
    }
    colors = ["lightgray", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"]

    encoder_names = report_df["encoder_name"].drop_duplicates().tolist()
    bar_width = 0.10
    group_gap = 0.24
    single_shift = 0.04

    all_positions = []
    for encoder_index, encoder_name in enumerate(encoder_names):
        subset = report_df[report_df["encoder_name"] == encoder_name].copy()
        subset["run_label"] = pd.Categorical(subset["run_label"], categories=ordered_runs, ordered=True)
        subset = subset.sort_values("run_label")

        offset = encoder_index * (len(ordered_runs) * bar_width + group_gap)
        x_positions = [offset + run_index * bar_width for run_index in range(len(ordered_runs))]
        x_positions[0] -= single_shift
        all_positions.append(x_positions)

        for run_index, (_, row) in enumerate(subset.iterrows()):
            ax.bar(
                x_positions[run_index],
                row["mean_accuracy_pct"],
                width=bar_width,
                yerr=row["std_accuracy_pct"],
                capsize=4,
                color=colors[run_index],
                edgecolor="black",
                linewidth=1.0,
                hatch="//" if run_index == 0 else None,
                label=display_labels[row["run_label"]] if encoder_index == 0 else None,
            )

        single_height = subset.iloc[0]["mean_accuracy_pct"] + subset.iloc[0]["std_accuracy_pct"]
        for run_index, (_, row) in enumerate(subset.iloc[1:].iterrows(), start=1):
            p_value = row["p_value_vs_single"]
            if pd.notna(p_value) and p_value < 0.05:
                multi_height = row["mean_accuracy_pct"] + row["std_accuracy_pct"]
                bracket_base = max(single_height, multi_height) + 2.0 + (run_index - 1) * 3.0
                add_significance_bracket(x_positions[0], x_positions[run_index], bracket_base, h=1.5, text="*", ax=ax)

    centers = [(positions[0] + positions[-1]) / 2.0 for positions in all_positions]
    xtick_labels = ["WaveNet-style" if name == "MyWaveNet" else name for name in encoder_names]

    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_xticks(centers, xtick_labels, fontsize=14)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(*y_limit)
    ax.legend(loc="upper left")
    return ax


def plot_patient_power_scatter(
    df: pd.DataFrame,
    *,
    x_col: str = "log_power",
    y_col: str = "log_rms_delta_r",
    patient_col: str = "patient_id",
    ax: Optional[plt.Axes] = None,
) -> Mapping[str, float]:
    """Scatter plot of log power vs. motion-trace RMS, colored by patient."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    patient_ids = sorted(df[patient_col].dropna().unique())
    cmap = plt.cm.get_cmap("tab10", len(patient_ids))
    for index, patient_id in enumerate(patient_ids):
        mask = df[patient_col] == patient_id
        ax.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            label=f"ET 0{patient_id}",
            s=20,
            alpha=0.7,
            color=cmap(index),
        )

    x_values = df[x_col].to_numpy(dtype=float)
    y_values = df[y_col].to_numpy(dtype=float)
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_valid = x_values[valid_mask]
    y_valid = y_values[valid_mask]

    pearson_r, pearson_p = pearsonr(x_valid, y_valid)
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, linewidth=2, color="lightgray", linestyle="--", label="Linear fit")

    ax.set_xticks([-5, -4, -3, -2, -1])
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Log Power", fontsize=16)
    ax.set_ylabel("Log RMS dr", fontsize=16)
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {pearson_r:.3f}\np = {pearson_p:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=16,
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return {"pearson_r": float(pearson_r), "pearson_p": float(pearson_p)}


def plot_patient_task_scatter_grid(
    df: pd.DataFrame,
    *,
    x_col: str = "log_power",
    y_col: str = "log_rms_delta_r",
    patient_col: str = "patient_id",
    task_col: str = "task",
    task_name_map: Optional[Mapping[str, str]] = None,
    task_order: Sequence[str] = ("SPN", "MDT", "RTN"),
    save_path=None,
) -> plt.Figure:
    """Scatter grid showing patient-wise task clusters with per-patient fits."""
    if task_name_map is None:
        task_name_map = {"Spiral": "SPN", "Maze": "RTN", "Multidirect": "MDT"}

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    patient_ids = sorted(df[patient_col].dropna().unique())

    for index, patient_id in enumerate(patient_ids):
        ax = axes[index]
        row = index // 3
        col = index % 3

        patient_mask = df[patient_col] == patient_id
        available_tasks = df.loc[patient_mask, task_col].dropna().unique().tolist()
        tasks = sorted(
            [task for task in available_tasks if task in task_name_map],
            key=lambda task: task_order.index(task_name_map[task]),
        )
        cmap = plt.cm.get_cmap("tab10", len(tasks))

        for task_index, task_name in enumerate(tasks):
            mask = patient_mask & (df[task_col] == task_name)
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                label=task_name_map[task_name],
                s=20,
                alpha=0.7,
                color=cmap(task_index),
            )

        x_values = df.loc[patient_mask, x_col].to_numpy(dtype=float)
        y_values = df.loc[patient_mask, y_col].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_valid = x_values[valid_mask]
        y_valid = y_values[valid_mask]

        if len(x_valid) > 1:
            pearson_r, pearson_p = pearsonr(x_valid, y_valid)
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                linewidth=2,
                color="gray",
                linestyle="--",
                label="Linear fit" if (row == 0 and col == 2) else None,
            )
            ax.text(
                0.05,
                0.95,
                f"Pearson r = {pearson_r:.3f}\np = {pearson_p:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=15,
            )

        ax.set_xlim(-5, -1)
        ax.set_ylim(0, 2)
        ax.set_xticks([-5, -4, -3, -2, -1])
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title(f"ET 0{patient_id}")
        ax.grid(True, alpha=0.3)

        if row != 2:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if col != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row == 0 and col == 2:
            handles, labels = ax.get_legend_handles_labels()
            ordered_handles = []
            ordered_labels = []
            for label in task_order:
                if label in labels:
                    ordered_handles.append(handles[labels.index(label)])
                    ordered_labels.append(label)
            if "Linear fit" in labels:
                ordered_handles.append(handles[labels.index("Linear fit")])
                ordered_labels.append("Linear fit")
            ax.legend(ordered_handles, ordered_labels, loc="lower right", fontsize=14)
        else:
            legend = ax.legend()
            if legend is not None:
                legend.set_visible(False)

    for ax in axes[len(patient_ids):]:
        ax.axis("off")

    fig.supxlabel("Log Power", fontsize=20)
    fig.supylabel("Log RMS dr", fontsize=20)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


def plot_patient_scatter_with_calibration_band(
    df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    *,
    x_col: str = "log_power",
    y_col: str = "log_rms_delta_r",
    patient_col: str = "patient_id",
    task_col: str = "task",
    task_name_map: Optional[Mapping[str, str]] = None,
) -> plt.Figure:
    """Scatter grid with calibration mean and one-standard-deviation bands."""
    if task_name_map is None:
        task_name_map = {"Spiral": "SPN", "Maze": "RTN", "Multidirect": "MDT"}

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    patient_ids = sorted(df[patient_col].dropna().unique())
    task_order = ("SPN", "MDT", "RTN")

    calibration_groups = {
        patient_id: calibration_df.loc[calibration_df[patient_col] == patient_id, x_col].to_numpy(dtype=float)
        for patient_id in patient_ids
    }

    for index, patient_id in enumerate(patient_ids):
        ax = axes[index]
        patient_mask = df[patient_col] == patient_id
        available_tasks = df.loc[patient_mask, task_col].dropna().unique().tolist()
        tasks = sorted(
            [task for task in available_tasks if task in task_name_map],
            key=lambda task: task_order.index(task_name_map[task]),
        )
        cmap = plt.cm.get_cmap("tab10", len(tasks))

        for task_index, task_name in enumerate(tasks):
            mask = patient_mask & (df[task_col] == task_name)
            ax.scatter(
                df.loc[mask, x_col],
                df.loc[mask, y_col],
                label=task_name_map[task_name],
                s=20,
                alpha=0.7,
                color=cmap(task_index),
            )

        calibration_values = calibration_groups.get(patient_id, np.array([], dtype=float))
        calibration_values = calibration_values[np.isfinite(calibration_values)]
        if calibration_values.size:
            mean_value = float(calibration_values.mean())
            std_value = float(calibration_values.std(ddof=0))
            if std_value > 0:
                ax.axvspan(mean_value - std_value, mean_value + std_value, color="lightgray", alpha=0.3, zorder=0)
            ax.axvline(mean_value, linestyle=":", linewidth=2, color="black", alpha=0.9)

        x_values = df.loc[patient_mask, x_col].to_numpy(dtype=float)
        y_values = df.loc[patient_mask, y_col].to_numpy(dtype=float)
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_valid = x_values[valid_mask]
        y_valid = y_values[valid_mask]
        if len(x_valid) > 1:
            pearson_r, pearson_p = pearsonr(x_valid, y_valid)
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, linewidth=2, color="lightgray", linestyle="--", label="Linear fit" if index == 0 else None)
            ax.text(
                0.05,
                0.95,
                f"Pearson r = {pearson_r:.3f}\np = {pearson_p:.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )

        ax.set_xlim(-5, -1)
        ax.set_ylim(0, 2)
        ax.set_xticks([-5, -4, -3, -2, -1])
        ax.set_yticks([0, 0.5, 1, 1.5, 2])
        ax.set_xlabel("Log Power")
        ax.set_ylabel("Log RMS dr")
        ax.set_title(f"ET 0{patient_id}")
        ax.grid(True, alpha=0.3)

        if index == 0:
            handles, labels = ax.get_legend_handles_labels()
            ordered_handles = []
            ordered_labels = []
            for label in task_order:
                if label in labels:
                    ordered_handles.append(handles[labels.index(label)])
                    ordered_labels.append(label)
            if "Linear fit" in labels:
                ordered_handles.append(handles[labels.index("Linear fit")])
                ordered_labels.append("Linear fit")
            ax.legend(ordered_handles, ordered_labels, loc="upper right")
        else:
            legend = ax.legend()
            if legend is not None:
                legend.set_visible(False)

    for ax in axes[len(patient_ids):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_log_power_boxplot(
    df: pd.DataFrame,
    *,
    category_col: str,
    value_col: str = "log_power",
    category_order: Optional[Sequence[float]] = None,
    label_transform=None,
    x_label: str,
    y_label: str = "Log Power",
    y_ticks: Sequence[float] = (-5, -4, -3, -2),
    y_limits: tuple[float, float] = (-5.1, -1.1),
    save_path=None,
) -> plt.Axes:
    """Draw the paper-style boxplot used for supplementary log-power figures."""
    df_plot = df[[category_col, value_col]].dropna().copy()
    if category_order is None:
        category_order = sorted(pd.to_numeric(df_plot[category_col], errors="coerce").dropna().unique())
    df_plot[category_col] = pd.Categorical(df_plot[category_col], categories=category_order, ordered=True)

    groups = [
        group[value_col].to_numpy(dtype=float)
        for _, group in df_plot.sort_values(category_col).groupby(category_col)
    ]
    labels = []
    for category in df_plot[category_col].cat.categories:
        if pd.isna(category):
            continue
        label = category
        if label_transform is not None:
            label = label_transform(category)
        labels.append(str(label))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        groups,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        whis=1.5,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(color="orange", linewidth=2.0),
        meanprops=dict(marker="^", markersize=7, markerfacecolor="green", markeredgecolor="green"),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.set_ylim(*y_limits)
    ax.set_yticks(list(y_ticks))
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return ax


def plot_general_vs_calibration_distribution(
    general_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    value_col: str = "log_power",
) -> tuple[pd.DataFrame, plt.Axes]:
    """Compare general-task log power with calibration log power per patient."""
    general = general_df.copy()
    calibration = calibration_df.copy()
    general["Task_Type"] = "SPN/RTN/MDT"
    calibration["Task_Type"] = "Bean-transfer"
    combined = pd.concat([general, calibration], axis=0, ignore_index=True)

    rows = []
    patient_ids = sorted(combined[patient_col].dropna().unique())
    for patient_id in patient_ids:
        general_values = general.loc[general[patient_col] == patient_id, value_col].dropna()
        calibration_values = calibration.loc[calibration[patient_col] == patient_id, value_col].dropna()
        if len(general_values) > 1 and len(calibration_values) > 1:
            _, p_value = mannwhitneyu(
                general_values,
                calibration_values,
                alternative="two-sided",
            )
            result_label = "** Diff **" if p_value < 0.05 else "Same"
        else:
            p_value = np.nan
            result_label = "Not enough data"
        rows.append(
            {
                "patient_id": patient_id,
                "n_general": int(len(general_values)),
                "n_calibration": int(len(calibration_values)),
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "result": result_label,
            }
        )

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        data=combined,
        x=patient_col,
        y=value_col,
        hue="Task_Type",
        palette="pastel",
        showfliers=False,
        width=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=combined,
        x=patient_col,
        y=value_col,
        hue="Task_Type",
        palette="dark:gray",
        dodge=True,
        jitter=True,
        alpha=0.7,
        size=4,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.set_yticks([-5, -4, -3, -2])
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(handles[:2], labels[:2], title="Task Type", loc="upper right")
    ax.set_ylabel("Log Power", fontsize=16)
    ax.set_xlabel("ET Participants", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return pd.DataFrame(rows), ax


def predict_calibration_labels_from_gmm(
    calibration_df: pd.DataFrame,
    gmm,
    *,
    power_col: str = "log_power",
    source_col: str = "normalized_power",
    file_col: str = "file",
    file_suffix: str = "calibration_1.txt",
    merge_lowest_cluster_into_one: bool = True,
) -> pd.DataFrame:
    """Project calibration samples onto ordered GMM components."""
    output = calibration_df.copy()
    if power_col not in output.columns:
        if source_col not in output.columns:
            raise KeyError(f"Expected either '{power_col}' or '{source_col}' in calibration dataframe.")
        output[power_col] = np.log10(output[source_col])

    x_values = output[power_col].to_numpy(dtype=float).reshape(-1, 1)
    raw_labels = gmm.predict(x_values)
    sorted_indices = np.argsort(gmm.means_.ravel())
    index_map = {original_index: rank for rank, original_index in enumerate(sorted_indices)}
    output["gmm_pred"] = [index_map[label] for label in raw_labels]
    if merge_lowest_cluster_into_one:
        output.loc[output["gmm_pred"] == 0, "gmm_pred"] = 1
    return output.loc[output[file_col].str.endswith(file_suffix)].copy()


def plot_mean_severity_by_tetras(
    table_df: pd.DataFrame,
    *,
    tetras_col: str = "TETRAS",
    severity_col: str = "Mean Sev",
    save_path=None,
) -> plt.Axes:
    """Plot median mean severity with MAD bars for each TETRAS score."""
    grouped = table_df.groupby(tetras_col)[severity_col]
    medians = grouped.median()
    mads = grouped.apply(median_absolute_deviation)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        medians.index.to_numpy(dtype=float),
        medians.to_numpy(dtype=float),
        yerr=mads.to_numpy(dtype=float),
        fmt="o-",
        capsize=5,
        ecolor="gray",
        color="gray",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=8,
    )
    ax.set_ylim(1, 3)
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("TETRAS score", fontsize=16)
    ax.set_ylabel("Mean Sev", fontsize=16)
    fig.tight_layout()
    _save_if_requested(fig, save_path)
    return ax
