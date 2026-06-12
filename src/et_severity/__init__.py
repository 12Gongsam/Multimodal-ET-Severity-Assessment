"""Public package API."""

from importlib import import_module

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_TARGET_PER_CLASS,
    DEFAULT_USECOLS,
    FS,
    HOP,
    NUM_WORKERS,
    SEED,
    SEG_LEN,
)
from .data import build_LOSO_loaders, build_manifest, split_and_delete_multidirect_fs
from .models import (
    EncoderConfig,
    MIL_MultiModal,
    MIL_Single,
    MultimodalModelConfig,
    SingleModelConfig,
    build_encoder,
    build_multimodal_model,
    build_single_modality_model,
)
from .training import (
    TrainingConfig,
    TrainingRun,
    evaluate_multimodal_model,
    evaluate_single_model,
    fit_multimodal,
    fit_single_modality,
    prepare_training_data,
    run_multimodal_severity_loso,
    run_single_modality_loso,
    summarize_fold_results,
    train_multimodal_model,
    train_single_model,
)
from .utils import set_seed

_LAZY_EXPORTS = {
    "evaluation_to_prediction_frame": ".analysis",
    "compute_patient_level_metrics": ".analysis",
    "summarize_metric_columns": ".analysis",
    "compare_ordinal_ratings": ".analysis",
    "median_absolute_deviation": ".analysis",
    "summarize_prediction_file": ".analysis",
    "summarize_prediction_collection": ".analysis",
    "summarize_multimodal_prediction_runs": ".analysis",
    "summarize_single_modality_prediction_runs": ".analysis",
    "build_single_vs_multimodal_accuracy_report": ".analysis",
    "build_session_severity_table": ".analysis",
    "build_session_severity_table_from_csv": ".analysis",
    "save_table_csv": ".analysis",
    "plot_single_vs_multimodal_accuracy": ".visualization",
    "plot_gaussian_mixture_fit": ".visualization",
    "plot_patient_power_scatter": ".visualization",
    "plot_patient_task_scatter_grid": ".visualization",
    "plot_patient_scatter_with_calibration_band": ".visualization",
    "plot_log_power_boxplot": ".visualization",
    "plot_general_vs_calibration_distribution": ".visualization",
    "predict_calibration_labels_from_gmm": ".visualization",
    "plot_mean_severity_by_tetras": ".visualization",
}


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "SEED",
    "FS",
    "SEG_LEN",
    "HOP",
    "NUM_WORKERS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE",
    "DEFAULT_TARGET_PER_CLASS",
    "DEFAULT_USECOLS",
    "set_seed",
    "split_and_delete_multidirect_fs",
    "build_manifest",
    "build_LOSO_loaders",
    "EncoderConfig",
    "SingleModelConfig",
    "MultimodalModelConfig",
    "MIL_MultiModal",
    "MIL_Single",
    "build_encoder",
    "build_single_modality_model",
    "build_multimodal_model",
    "TrainingConfig",
    "TrainingRun",
    "train_single_model",
    "train_multimodal_model",
    "fit_single_modality",
    "fit_multimodal",
    "evaluate_single_model",
    "evaluate_multimodal_model",
    "prepare_training_data",
    "run_multimodal_severity_loso",
    "run_single_modality_loso",
    "summarize_fold_results",
    "compute_patient_level_metrics",
    "evaluation_to_prediction_frame",
    "summarize_metric_columns",
    "compare_ordinal_ratings",
    "median_absolute_deviation",
    "summarize_prediction_file",
    "summarize_prediction_collection",
    "summarize_multimodal_prediction_runs",
    "summarize_single_modality_prediction_runs",
    "build_single_vs_multimodal_accuracy_report",
    "build_session_severity_table",
    "build_session_severity_table_from_csv",
    "save_table_csv",
    "plot_single_vs_multimodal_accuracy",
    "plot_gaussian_mixture_fit",
    "plot_patient_power_scatter",
    "plot_patient_task_scatter_grid",
    "plot_patient_scatter_with_calibration_band",
    "plot_log_power_boxplot",
    "plot_general_vs_calibration_distribution",
    "predict_calibration_labels_from_gmm",
    "plot_mean_severity_by_tetras",
]
