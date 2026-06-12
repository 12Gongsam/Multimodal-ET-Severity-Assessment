"""Public package API."""

from importlib import import_module

from .config import DEFAULT_DEVICE, HOP, NUM_WORKERS, SEED, SEG_LEN
from .data import build_LOSO_loaders, build_manifest
from .models import (
    EncoderConfig,
    JointInstanceAttentionConfig,
    MultimodalModelConfig,
    ResidualMultimodalModelConfig,
    SingleModelConfig,
)
from .training import (
    TrainingConfig,
    TrainingRun,
    calculate_classification_metrics,
    fit_multimodal,
    fit_single_modality,
    run_loso_cv,
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
    "SEG_LEN",
    "HOP",
    "NUM_WORKERS",
    "DEFAULT_DEVICE",
    "set_seed",
    "build_manifest",
    "build_LOSO_loaders",
    "EncoderConfig",
    "SingleModelConfig",
    "MultimodalModelConfig",
    "JointInstanceAttentionConfig",
    "ResidualMultimodalModelConfig",
    "TrainingConfig",
    "TrainingRun",
    "fit_single_modality",
    "fit_multimodal",
    "calculate_classification_metrics",
    "run_loso_cv",
    *_LAZY_EXPORTS,
]
