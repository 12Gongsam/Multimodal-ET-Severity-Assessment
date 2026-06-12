"""Colab-friendly training entry point.

Example:
    python colab_train.py --experiments multimodal --patient-ids 1 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from et_severity.config import DEFAULT_DEVICE
from et_severity.data import build_LOSO_loaders, build_manifest
from et_severity.models import (
    EncoderConfig,
    MultimodalModelConfig,
    SingleModelConfig,
)
from et_severity.training import (
    TrainingConfig,
    run_multimodal_severity_loso,
    run_single_modality_loso,
)


def _json_object(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed


def _parse_patient_ids(values: Optional[Sequence[int]]) -> Optional[list[int]]:
    if values is None:
        return None
    return [int(value) for value in values]


def _make_training_config(args) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.patience,
        optimizer=args.optimizer,
        monitor=args.monitor,
        use_scheduler=not args.no_scheduler,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_cooldown=args.scheduler_cooldown,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_threshold=args.scheduler_threshold,
    )


def _save_run_outputs(
    run_dir: Path,
    *,
    results: pd.DataFrame,
    summary: Dict[str, float],
    histories: Dict[str, Dict[str, list]],
    run_config: Dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_dir / "fold_results.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "histories.json").write_text(
        json.dumps(histories, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )


def _build_loaders(manifest, args, *, modality: str):
    return build_LOSO_loaders(
        manifest,
        target_per_class=args.target_per_class,
        filter_task=args.filter_task,
        modality=modality,
        patient_ids=_parse_patient_ids(args.patient_ids),
        batch_size=args.batch_size,
        seg_len=args.segment_length,
        hop=args.segment_hop,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
    )


def _run_multimodal(manifest, args, training_config, output_root: Path):
    model_config = MultimodalModelConfig(
        acc_encoder=EncoderConfig(
            args.acc_encoder,
            args.acc_encoder_params,
        ),
        traj_encoder=EncoderConfig(
            args.traj_encoder,
            args.traj_encoder_params,
        ),
        num_classes=args.severity_classes,
        d_model=args.d_model,
        cross_attention_heads=args.cross_attention_heads,
        cross_attention_dropout=args.cross_attention_dropout,
        time_dropout=args.time_dropout,
        mil_attn_dim=args.mil_attention_dim,
        time_pool=args.time_pool,
        seq_len=args.segment_length,
    )
    run_dir = output_root / "multimodal"
    loaders = _build_loaders(manifest, args, modality="multimodal")
    results, summary, histories = run_multimodal_severity_loso(
        loaders,
        model_config=model_config,
        training_config=training_config,
        device=DEFAULT_DEVICE,
        checkpoint_dir=run_dir / "checkpoints",
        seed=args.seed,
    )
    _save_run_outputs(
        run_dir,
        results=results,
        summary=summary,
        histories=histories,
        run_config={
            "experiment": "multimodal",
            "model": asdict(model_config),
            "training": asdict(training_config),
            "data": _data_config(args, "multimodal"),
        },
    )
    return results, summary


def _run_single(
    manifest,
    args,
    training_config,
    output_root: Path,
    *,
    modality: str,
):
    num_classes = (
        args.severity_classes if args.target == "severity" else args.task_classes
    )
    model_config = SingleModelConfig(
        encoder=EncoderConfig(
            args.single_encoder,
            args.single_encoder_params,
        ),
        num_classes=num_classes,
        d_model=args.d_model,
        time_dropout=args.time_dropout,
        mil_attn_dim=args.mil_attention_dim,
        mil_dropout=args.mil_dropout,
        seq_len=args.segment_length,
    )
    run_dir = output_root / f"{modality}_{args.target}"
    loaders = _build_loaders(manifest, args, modality=modality)
    results, summary, histories = run_single_modality_loso(
        loaders,
        model_config=model_config,
        modality=modality,
        target=args.target,
        training_config=training_config,
        device=DEFAULT_DEVICE,
        checkpoint_dir=run_dir / "checkpoints",
        seed=args.seed,
    )
    _save_run_outputs(
        run_dir,
        results=results,
        summary=summary,
        histories=histories,
        run_config={
            "experiment": modality,
            "target": args.target,
            "model": asdict(model_config),
            "training": asdict(training_config),
            "data": _data_config(args, modality),
        },
    )
    return results, summary


def _data_config(args, modality: str) -> Dict[str, Any]:
    return {
        "data_root": str(args.data_root),
        "label_csv": str(args.label_csv),
        "target_column": args.target_column,
        "modality": modality,
        "filter_task": args.filter_task,
        "target_per_class": args.target_per_class,
        "batch_size": args.batch_size,
        "segment_length": args.segment_length,
        "segment_hop": args.segment_hop,
        "num_workers": args.num_workers,
        "patient_ids": _parse_patient_ids(args.patient_ids),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ET severity models from /content/data in Colab."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("multimodal", "acc", "traj"),
        default=("multimodal",),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/content/data"))
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=Path("/content/data/relabel_md_k5.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/content/et_severity_runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--target-column", default="target_k5")
    parser.add_argument("--target", choices=("severity", "task"), default="severity")
    parser.add_argument("--filter-task", default=None)
    parser.add_argument("--patient-ids", nargs="+", type=int, default=None)
    parser.add_argument("--target-per-class", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-length", type=int, default=512)
    parser.add_argument("--segment-hop", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")

    parser.add_argument("--severity-classes", type=int, default=4)
    parser.add_argument("--task-classes", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--mil-attention-dim", type=int, default=64)
    parser.add_argument("--time-dropout", type=float, default=0.1)
    parser.add_argument("--mil-dropout", type=float, default=0.1)

    parser.add_argument("--single-encoder", default="MyWaveNet")
    parser.add_argument(
        "--single-encoder-params",
        type=_json_object,
        default={
            "residual_channels": 128,
            "skip_channels": 128,
            "n_stacks": 2,
        },
    )
    parser.add_argument("--acc-encoder", default="LSTM")
    parser.add_argument(
        "--acc-encoder-params",
        type=_json_object,
        default={"hidden_size": 128, "num_layers": 2},
    )
    parser.add_argument("--traj-encoder", default="ResNet18")
    parser.add_argument(
        "--traj-encoder-params",
        type=_json_object,
        default={"feature_dim": 128},
    )
    parser.add_argument("--cross-attention-heads", type=int, default=8)
    parser.add_argument("--cross-attention-dropout", type=float, default=0.1)
    parser.add_argument("--time-pool", choices=("attn", "gap"), default="attn")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--monitor", choices=("loss", "macro_f1"), default="loss")
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-cooldown", type=int, default=1)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_root}")
    if not args.label_csv.is_file():
        raise FileNotFoundError(f"Label CSV not found: {args.label_csv}")

    print(f"Device: {DEFAULT_DEVICE}")
    print(f"Data root: {args.data_root}")
    print(f"Label CSV: {args.label_csv}")
    manifest = build_manifest(
        args.label_csv,
        args.data_root,
        target_col=args.target_column,
    )
    if manifest.empty:
        raise RuntimeError(
            "The training manifest is empty. Check label CSV paths and sensor files."
        )
    print(
        f"Manifest rows: {len(manifest)} | "
        f"patients: {manifest['patient_id'].nunique()}"
    )

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / run_name
    training_config = _make_training_config(args)

    for experiment in args.experiments:
        print(f"\n=== Experiment: {experiment} ===")
        if experiment == "multimodal":
            _, summary = _run_multimodal(
                manifest, args, training_config, output_root
            )
        else:
            _, summary = _run_single(
                manifest,
                args,
                training_config,
                output_root,
                modality=experiment,
            )
        print(json.dumps(summary, indent=2))

    print(f"\nOutputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
