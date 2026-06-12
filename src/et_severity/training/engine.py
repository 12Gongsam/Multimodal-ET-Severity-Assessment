"""Separate training engines for single- and multimodal MIL models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: Optional[float] = 1.0
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    monitor: str = "loss"
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_cooldown: int = 1
    scheduler_min_lr: float = 1e-6
    scheduler_threshold: float = 1e-3
    scheduler_threshold_mode: str = "rel"
    scheduler_eps: float = 1e-8


def _metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, object]:
    confusion = confusion.float()
    class_count = confusion.size(0)
    if class_count == 0:
        return {"per_class_acc": [], "macro_f1": 0.0}

    support = confusion.sum(dim=1)
    prediction_count = confusion.sum(dim=0)
    true_positive = confusion.diag()
    present = support > 0

    per_class_acc = torch.full((class_count,), float("nan"))
    per_class_acc[present] = true_positive[present] / support[present]

    precision = torch.zeros(class_count)
    recall = torch.zeros(class_count)
    precision[prediction_count > 0] = (
        true_positive[prediction_count > 0]
        / prediction_count[prediction_count > 0]
    )
    recall[present] = true_positive[present] / support[present]

    denominator = precision + recall
    f1 = torch.zeros(class_count)
    nonzero = denominator > 0
    f1[nonzero] = (
        2 * precision[nonzero] * recall[nonzero] / denominator[nonzero]
    )
    macro_f1 = float(f1[present].mean().item()) if present.any() else 0.0
    return {
        "per_class_acc": per_class_acc.tolist(),
        "macro_f1": macro_f1,
    }


class _MetricAccumulator:
    def __init__(self, *, collect_outputs: bool = False):
        self.total_loss = 0.0
        self.batches = 0
        self.correct = 0
        self.total = 0
        self.confusion: Optional[torch.Tensor] = None
        self.collect_outputs = collect_outputs
        self.logits = []
        self.labels = []
        self.meta = []

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        meta=None,
    ) -> None:
        logits_cpu = logits.detach().cpu()
        predictions = logits_cpu.argmax(dim=-1)
        labels = labels.detach().cpu()
        class_count = logits.size(-1)
        if self.confusion is None:
            self.confusion = torch.zeros(
                class_count, class_count, dtype=torch.long
            )
        indices = labels.long() * class_count + predictions.long()
        self.confusion += torch.bincount(
            indices, minlength=class_count * class_count
        ).reshape(class_count, class_count)
        self.total_loss += float(loss.detach().item())
        self.batches += 1
        self.correct += int((predictions == labels).sum().item())
        self.total += int(labels.numel())
        if self.collect_outputs:
            self.logits.append(logits_cpu)
            self.labels.append(labels)
            if meta is not None:
                self.meta.extend(meta)

    def result(self) -> Dict[str, object]:
        confusion = (
            self.confusion
            if self.confusion is not None
            else torch.zeros(0, 0, dtype=torch.long)
        )
        metrics = _metrics_from_confusion(confusion)
        result = {
            "loss": self.total_loss / max(1, self.batches),
            "acc": self.correct / max(1, self.total),
            "macro_f1": metrics["macro_f1"],
            "per_class_acc": metrics["per_class_acc"],
            "conf": confusion,
        }
        if self.collect_outputs:
            logits = (
                torch.cat(self.logits)
                if self.logits
                else torch.empty((0, confusion.size(0)))
            )
            labels = (
                torch.cat(self.labels)
                if self.labels
                else torch.empty(0, dtype=torch.long)
            )
            result.update(
                {
                    "y_true": labels,
                    "y_pred": (
                        logits.argmax(dim=-1)
                        if logits.size(-1) > 0
                        else torch.empty(0, dtype=torch.long)
                    ),
                    "logits": logits,
                    "probabilities": torch.softmax(logits, dim=-1),
                    "meta": list(self.meta),
                }
            )
        return result


def _apply_metric_fns(
    result: Dict[str, object],
    metric_fns: Optional[Mapping[str, Callable[[Dict[str, object]], object]]],
) -> Dict[str, object]:
    if metric_fns:
        for name, metric_fn in metric_fns.items():
            if name in result:
                raise ValueError(f"Metric name {name!r} is already reserved.")
            result[name] = metric_fn(result)
    return result


def _target_from_batch(batch, target: str, device: torch.device):
    if target == "severity":
        return batch["y"].to(device, non_blocking=True)
    if target == "task":
        return batch["y_task"].to(device, non_blocking=True)
    raise ValueError("target must be 'severity' or 'task'")


def _single_epoch(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    modality: str,
    target: str,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: Optional[float],
    class_weights: Optional[torch.Tensor],
    collect_outputs: bool = False,
) -> Dict[str, object]:
    if modality not in {"acc", "traj"}:
        raise ValueError("modality must be 'acc' or 'traj'")

    is_training = optimizer is not None
    model.train(is_training)
    accumulator = _MetricAccumulator(collect_outputs=collect_outputs)
    progress = tqdm(
        loader,
        desc="Train(single)" if is_training else "Valid(single)",
        leave=False,
    )
    input_key = "acc_bag" if modality == "acc" else "traj_bag"
    weights = (
        None
        if class_weights is None
        else class_weights.to(device=device, dtype=torch.float32)
    )

    for batch in progress:
        inputs = batch[input_key].permute(0, 1, 3, 2).to(
            device, non_blocking=True
        )
        mask = batch["mask"].to(device, non_blocking=True)
        labels = _target_from_batch(batch, target, device)

        with torch.set_grad_enabled(is_training):
            logits, _ = model(inputs, mask=mask)
            loss = F.cross_entropy(logits, labels, weight=weights)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        accumulator.update(loss, logits, labels, batch.get("meta"))
        current = accumulator.result()
        progress.set_postfix(
            loss=f"{current['loss']:.4f}",
            acc=f"{current['acc']:.3f}",
            macro_f1=f"{current['macro_f1']:.3f}",
        )

    return accumulator.result()


def _multimodal_epoch(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: Optional[float],
    class_weights: Optional[torch.Tensor],
    collect_outputs: bool = False,
) -> Dict[str, object]:
    is_training = optimizer is not None
    model.train(is_training)
    accumulator = _MetricAccumulator(collect_outputs=collect_outputs)
    progress = tqdm(
        loader,
        desc="Train(multimodal)" if is_training else "Valid(multimodal)",
        leave=False,
    )
    weights = (
        None
        if class_weights is None
        else class_weights.to(device=device, dtype=torch.float32)
    )

    for batch in progress:
        acc = batch["acc_bag"].permute(0, 1, 3, 2).to(
            device, non_blocking=True
        )
        traj = batch["traj_bag"].permute(0, 1, 3, 2).to(
            device, non_blocking=True
        )
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["y"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            logits, _ = model(acc, traj, mask=mask)
            loss = F.cross_entropy(logits, labels, weight=weights)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        accumulator.update(loss, logits, labels, batch.get("meta"))
        current = accumulator.result()
        progress.set_postfix(
            loss=f"{current['loss']:.4f}",
            acc=f"{current['acc']:.3f}",
            macro_f1=f"{current['macro_f1']:.3f}",
        )

    return accumulator.result()


def _build_optimizer(model: nn.Module, config: TrainingConfig):
    optimizer_name = config.optimizer.strip().lower()
    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError("optimizer must be 'adam' or 'adamw'")


def _fit(
    model: nn.Module,
    train_epoch: Callable[[Optional[torch.optim.Optimizer]], Dict[str, object]],
    valid_epoch: Callable[[Optional[torch.optim.Optimizer]], Dict[str, object]],
    *,
    config: TrainingConfig,
) -> Dict[str, List[object]]:
    if config.monitor not in {"loss", "macro_f1"}:
        raise ValueError("monitor must be 'loss' or 'macro_f1'")

    optimizer = _build_optimizer(model, config)
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if config.monitor == "loss" else "max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            cooldown=config.scheduler_cooldown,
            min_lr=config.scheduler_min_lr,
            threshold=config.scheduler_threshold,
            threshold_mode=config.scheduler_threshold_mode,
            eps=config.scheduler_eps,
        )

    best_score = math.inf if config.monitor == "loss" else -math.inf
    best_state = None
    patience = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "train_per_class_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_macro_f1": [],
        "valid_per_class_acc": [],
        "lr": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(optimizer)
        valid_metrics = valid_epoch(None)
        score = valid_metrics[config.monitor]
        if scheduler is not None:
            scheduler.step(score)

        for prefix, metrics in (
            ("train", train_metrics),
            ("valid", valid_metrics),
        ):
            history[f"{prefix}_loss"].append(metrics["loss"])
            history[f"{prefix}_acc"].append(metrics["acc"])
            history[f"{prefix}_macro_f1"].append(metrics["macro_f1"])
            history[f"{prefix}_per_class_acc"].append(
                metrics["per_class_acc"]
            )
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"[Epoch {epoch:03d}] "
            f"TR loss {train_metrics['loss']:.4f} | "
            f"VA loss {valid_metrics['loss']:.4f} | "
            f"TR acc/F1 {train_metrics['acc']:.3f}/"
            f"{train_metrics['macro_f1']:.3f} | "
            f"VA acc/F1 {valid_metrics['acc']:.3f}/"
            f"{valid_metrics['macro_f1']:.3f} | "
            f"lr {history['lr'][-1]:.2e}"
        )

        improved = (
            score < best_score
            if config.monitor == "loss"
            else score > best_score
        )
        if improved:
            best_score = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {config.early_stopping_patience} epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_single_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    device: torch.device,
    modality: str,
    target: str,
    config: TrainingConfig = TrainingConfig(),
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, List[object]]:
    model.to(device)
    return _fit(
        model,
        lambda optimizer: _single_epoch(
            model,
            train_loader,
            device=device,
            modality=modality,
            target=target,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
            class_weights=class_weights,
        ),
        lambda optimizer: _single_epoch(
            model,
            valid_loader,
            device=device,
            modality=modality,
            target=target,
            optimizer=optimizer,
            grad_clip=None,
            class_weights=class_weights,
        ),
        config=config,
    )


def train_multimodal_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    device: torch.device,
    config: TrainingConfig = TrainingConfig(),
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, List[object]]:
    model.to(device)
    return _fit(
        model,
        lambda optimizer: _multimodal_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
            class_weights=class_weights,
        ),
        lambda optimizer: _multimodal_epoch(
            model,
            valid_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=None,
            class_weights=class_weights,
        ),
        config=config,
    )


@torch.no_grad()
def evaluate_single_model(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    modality: str,
    target: str,
    class_weights: Optional[torch.Tensor] = None,
    metric_fns: Optional[
        Mapping[str, Callable[[Dict[str, object]], object]]
    ] = None,
) -> Dict[str, object]:
    result = _single_epoch(
        model,
        loader,
        device=device,
        modality=modality,
        target=target,
        optimizer=None,
        grad_clip=None,
        class_weights=class_weights,
        collect_outputs=True,
    )
    return _apply_metric_fns(result, metric_fns)


@torch.no_grad()
def evaluate_multimodal_model(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
    metric_fns: Optional[
        Mapping[str, Callable[[Dict[str, object]], object]]
    ] = None,
) -> Dict[str, object]:
    result = _multimodal_epoch(
        model,
        loader,
        device=device,
        optimizer=None,
        grad_clip=None,
        class_weights=class_weights,
        collect_outputs=True,
    )
    return _apply_metric_fns(result, metric_fns)
