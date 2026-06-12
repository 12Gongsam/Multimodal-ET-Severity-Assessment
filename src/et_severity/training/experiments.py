"""High-level training functions for notebooks and Colab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Union

import torch
import torch.nn as nn

from ..config import DEFAULT_DEVICE
from ..models.mil_models import (
    MultimodalModelConfig,
    SingleModelConfig,
    build_multimodal_model,
    build_single_modality_model,
)
from ..utils import set_seed
from .engine import (
    TrainingConfig,
    evaluate_multimodal_model,
    evaluate_single_model,
    train_multimodal_model,
    train_single_model,
)

MetricFn = Callable[[Dict[str, object]], object]
DeviceLike = Union[str, torch.device]


@dataclass
class TrainingRun:
    model: nn.Module
    history: Dict[str, List[object]]
    metrics: Dict[str, object]
    model_config: Union[SingleModelConfig, MultimodalModelConfig]
    training_config: TrainingConfig


def _resolve_device(device: Optional[DeviceLike]) -> torch.device:
    if device is None:
        return DEFAULT_DEVICE
    return torch.device(device)


def fit_single_modality(
    train_loader,
    valid_loader,
    *,
    modality: str,
    model_config: SingleModelConfig,
    training_config: TrainingConfig = TrainingConfig(),
    target: str = "severity",
    device: Optional[DeviceLike] = None,
    class_weights: Optional[torch.Tensor] = None,
    metric_fns: Optional[Mapping[str, MetricFn]] = None,
    seed: int = 42,
) -> TrainingRun:
    """Build, train, and evaluate one single-modality model."""
    resolved_device = _resolve_device(device)
    set_seed(seed)
    model = build_single_modality_model(
        model_config,
        modality=modality,
    ).to(resolved_device)
    history = train_single_model(
        model,
        train_loader,
        valid_loader,
        device=resolved_device,
        modality=modality,
        target=target,
        config=training_config,
        class_weights=class_weights,
    )
    metrics = evaluate_single_model(
        model,
        valid_loader,
        device=resolved_device,
        modality=modality,
        target=target,
        class_weights=class_weights,
        metric_fns=metric_fns,
    )
    return TrainingRun(
        model=model,
        history=history,
        metrics=metrics,
        model_config=model_config,
        training_config=training_config,
    )


def fit_multimodal(
    train_loader,
    valid_loader,
    *,
    model_config: MultimodalModelConfig,
    training_config: TrainingConfig = TrainingConfig(),
    device: Optional[DeviceLike] = None,
    class_weights: Optional[torch.Tensor] = None,
    metric_fns: Optional[Mapping[str, MetricFn]] = None,
    seed: int = 42,
) -> TrainingRun:
    """Build, train, and evaluate one multimodal severity model."""
    resolved_device = _resolve_device(device)
    set_seed(seed)
    model = build_multimodal_model(model_config).to(resolved_device)
    history = train_multimodal_model(
        model,
        train_loader,
        valid_loader,
        device=resolved_device,
        config=training_config,
        class_weights=class_weights,
    )
    metrics = evaluate_multimodal_model(
        model,
        valid_loader,
        device=resolved_device,
        class_weights=class_weights,
        metric_fns=metric_fns,
    )
    return TrainingRun(
        model=model,
        history=history,
        metrics=metrics,
        model_config=model_config,
        training_config=training_config,
    )
