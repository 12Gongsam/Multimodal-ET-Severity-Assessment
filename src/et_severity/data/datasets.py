"""MIL datasets and LOSO dataloader builders."""

from __future__ import annotations

import collections
from functools import partial
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import HOP, NUM_WORKERS, SEG_LEN
from ..utils.reproducibility import set_seed
from .augmentations import BagRotator, InstancePermTimeW
from .preprocessing import build_segment_manifest, split_train_valid_by_patient

ACC_COLUMNS = ("accel_x", "accel_y", "accel_z")
TRAJ_COLUMNS = ("coor_x", "coor_y")
SUPPORTED_MODALITIES = {"acc", "traj", "multimodal"}


def _normalize_modality(modality: str) -> str:
    normalized = str(modality).strip().lower()
    if normalized not in SUPPORTED_MODALITIES:
        raise ValueError("modality must be 'acc', 'traj', or 'multimodal'")
    return normalized


def _columns_for_modality(modality: str) -> List[str]:
    modality = _normalize_modality(modality)
    if modality == "acc":
        return list(ACC_COLUMNS)
    if modality == "traj":
        return list(TRAJ_COLUMNS)
    return [*ACC_COLUMNS, *TRAJ_COLUMNS]


class _BaseMILLoaderPath:
    """Load one CSV-backed MIL bag using only the requested modalities."""

    def __init__(
        self,
        seg_df: pd.DataFrame,
        target_col: str = "target",
        *,
        modality: str = "multimodal",
        traj_div: float = 400.0,
        filter_task: Optional[str] = None,
    ):
        required = {"path", "start", "end", target_col, "task", "task_idx"}
        missing = required - set(seg_df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        self.df = seg_df.reset_index(drop=True).copy()
        self.target_col = target_col
        self.modality = _normalize_modality(modality)
        self.usecols = _columns_for_modality(self.modality)
        self.traj_div = float(traj_div)
        self.filter_task = filter_task

        if self.filter_task:
            self.df = self.df[self.df["task"] == self.filter_task].copy()
            print(
                f"[INFO] Dataset filtered for task: '{self.filter_task}'. "
                f"Total segments: {len(self.df)}"
            )

        self.df["_bagkey"] = self.df["path"].astype(str)
        self.bags = []
        for bagkey, group in self.df.groupby("_bagkey", sort=True):
            self.bags.append(
                {
                    "bagkey": bagkey,
                    "rows": group.sort_values("start"),
                    "target": int(group[self.target_col].mode().iloc[0]),
                    "task_idx": int(group["task_idx"].mode().iloc[0]),
                    "task": str(group["task"].mode().iloc[0]),
                    "patient_id": int(group["patient_id"].mode().iloc[0]),
                    "session": int(group["session"].mode().iloc[0]),
                    "date": str(group["date"].mode().iloc[0]),
                }
            )

    def load_bag(self, bag_idx: int) -> Dict[str, object]:
        bag = self.bags[bag_idx]
        rows = bag["rows"]
        path = str(bag["bagkey"])

        # A bag maps to one CSV. Read once, then slice all segments in memory.
        frame = pd.read_csv(path, usecols=self.usecols)
        acc_segments = []
        traj_segments = []

        for row in rows.itertuples(index=False):
            segment = frame.iloc[int(row.start):int(row.end)]
            if self.modality in {"acc", "multimodal"}:
                acc = segment.loc[:, list(ACC_COLUMNS)].to_numpy(dtype=np.float32).T
                acc_segments.append(torch.from_numpy(acc))
            if self.modality in {"traj", "multimodal"}:
                traj = segment.loc[:, list(TRAJ_COLUMNS)].to_numpy(dtype=np.float32).T
                traj_segments.append(torch.from_numpy(traj / self.traj_div))

        result: Dict[str, object] = {
            "target": int(bag["target"]),
            "task_target": int(bag["task_idx"]),
            "task": str(bag["task"]),
            "bagkey": path,
            "patient_id": int(bag["patient_id"]),
            "session": int(bag["session"]),
            "date": str(bag["date"]),
        }
        if acc_segments:
            result["acc_bag"] = torch.stack(acc_segments, dim=0)
        if traj_segments:
            result["traj_bag"] = torch.stack(traj_segments, dim=0)
        return result


class BalancedMILDatasetPath_RPT(Dataset):
    """Balanced training bags with optional synchronized augmentation."""

    def __init__(
        self,
        seg_train_df: pd.DataFrame,
        target_col: str = "target",
        bag_rot: Optional[BagRotator] = None,
        inst_perm_timew: Optional[InstancePermTimeW] = None,
        max_instances_per_bag: Optional[int] = None,
        print_stats: bool = True,
        target_per_class: Optional[int] = None,
        target_total_bags: Optional[int] = None,
        allow_downsample: bool = False,
        seed_balance: int = 2024,
        filter_task: Optional[str] = None,
        modality: str = "multimodal",
    ):
        self.base = _BaseMILLoaderPath(
            seg_train_df,
            target_col=target_col,
            filter_task=filter_task,
            modality=modality,
        )
        self.bag_rot = bag_rot
        self.inst_ptw = inst_perm_timew
        self.maxN = max_instances_per_bag

        self.orig_indices = list(range(len(self.base.bags)))
        targets = [bag["target"] for bag in self.base.bags]
        counter = collections.Counter(targets)
        self.classes = sorted(counter)
        self.orig_counts = {class_id: counter[class_id] for class_id in self.classes}
        max_count = max(counter.values()) if counter else 0

        if target_per_class is not None:
            desired = {class_id: int(target_per_class) for class_id in self.classes}
        elif target_total_bags is not None and self.classes:
            base_count, remainder = divmod(target_total_bags, len(self.classes))
            desired = {
                class_id: base_count + int(index < remainder)
                for index, class_id in enumerate(self.classes)
            }
        else:
            desired = {class_id: max_count for class_id in self.classes}

        if not allow_downsample:
            for class_id in self.classes:
                desired[class_id] = max(desired[class_id], self.orig_counts[class_id])

        per_class_indices = {
            class_id: [
                index
                for index, bag in enumerate(self.base.bags)
                if bag["target"] == class_id
            ]
            for class_id in self.classes
        }
        rng = np.random.default_rng(seed_balance)
        kept_indices = set(self.orig_indices)

        if allow_downsample:
            kept_indices = set()
            for class_id, class_indices in per_class_indices.items():
                keep_n = min(len(class_indices), desired[class_id])
                chosen = (
                    rng.choice(class_indices, size=keep_n, replace=False).tolist()
                    if keep_n < len(class_indices)
                    else class_indices
                )
                kept_indices.update(chosen)

        augmented_items = []
        for class_id, class_indices in per_class_indices.items():
            pool = [index for index in class_indices if index in kept_indices]
            needed = desired[class_id] - len(pool)
            if needed > 0 and pool:
                sampled = rng.integers(0, len(pool), size=needed)
                augmented_items.extend((pool[int(sample)], True) for sample in sampled)

        self.items = [(index, False) for index in sorted(kept_indices)] + augmented_items

        if print_stats and self.items:
            total_counts = collections.Counter(
                self.base.bags[index]["target"] for index, _ in self.items
            )
            print("[MIL Balance @ path-bag]")
            print(f"  Modality: {self.base.modality}")
            print(f"  Task Filter: {self.base.filter_task or 'None'}")
            print("  before :", self.orig_counts)
            print("  desired:", desired)
            print("  after  :", dict(total_counts), "| total:", len(self.items))

    def __len__(self):
        return len(self.items)

    def reset_random_state(self):
        if hasattr(self.bag_rot, "reset_random_state"):
            self.bag_rot.reset_random_state()
        if hasattr(self.inst_ptw, "reset_random_state"):
            self.inst_ptw.reset_random_state()

    def __getitem__(self, idx):
        bag_idx, is_augmented = self.items[idx]
        sample = self.base.load_bag(bag_idx)
        acc_bag = sample.get("acc_bag")
        traj_bag = sample.get("traj_bag")

        if is_augmented:
            if self.bag_rot is not None:
                acc_bag, traj_bag = self.bag_rot(acc_bag, traj_bag)
            if self.inst_ptw is not None:
                reference = acc_bag if acc_bag is not None else traj_bag
                augmented_acc = []
                augmented_traj = []
                for instance_index in range(reference.shape[0]):
                    acc_array = (
                        None
                        if acc_bag is None
                        else acc_bag[instance_index].cpu().numpy()
                    )
                    traj_array = (
                        None
                        if traj_bag is None
                        else traj_bag[instance_index].cpu().numpy()
                    )
                    acc_aug, traj_aug = self.inst_ptw(acc_array, traj_array)
                    if acc_aug is not None:
                        augmented_acc.append(torch.from_numpy(acc_aug).float())
                    if traj_aug is not None:
                        augmented_traj.append(torch.from_numpy(traj_aug).float())
                if augmented_acc:
                    acc_bag = torch.stack(augmented_acc, dim=0)
                if augmented_traj:
                    traj_bag = torch.stack(augmented_traj, dim=0)

        reference = acc_bag if acc_bag is not None else traj_bag
        if self.maxN is not None and reference.shape[0] > self.maxN:
            keep = torch.randperm(reference.shape[0])[:self.maxN]
            if acc_bag is not None:
                acc_bag = acc_bag[keep]
            if traj_bag is not None:
                traj_bag = traj_bag[keep]
            reference = reference[keep]

        result = {
            "mask": torch.ones(reference.shape[0], dtype=torch.bool),
            "target": sample["target"],
            "task_target": sample["task_target"],
            "meta": {
                "bagkey": sample["bagkey"],
                "N": reference.shape[0],
                "aug": bool(is_augmented),
                "task": sample["task"],
                "patient_id": sample["patient_id"],
                "session": sample["session"],
                "date": sample["date"],
            },
        }
        if acc_bag is not None:
            result["acc_bag"] = acc_bag
        if traj_bag is not None:
            result["traj_bag"] = traj_bag
        return result


class EvalMILDatasetPath(Dataset):
    def __init__(
        self,
        seg_valid_df: pd.DataFrame,
        target_col: str = "target",
        filter_task: Optional[str] = None,
        modality: str = "multimodal",
    ):
        self.base = _BaseMILLoaderPath(
            seg_valid_df,
            target_col=target_col,
            filter_task=filter_task,
            modality=modality,
        )

    def __len__(self):
        return len(self.base.bags)

    def __getitem__(self, index):
        sample = self.base.load_bag(index)
        reference = sample.get("acc_bag", sample.get("traj_bag"))
        result = {
            "mask": torch.ones(reference.shape[0], dtype=torch.bool),
            "target": sample["target"],
            "task_target": sample["task_target"],
            "meta": {
                "bagkey": sample["bagkey"],
                "N": reference.shape[0],
                "aug": False,
                "task": sample["task"],
                "patient_id": sample["patient_id"],
                "session": sample["session"],
                "date": sample["date"],
            },
        }
        if "acc_bag" in sample:
            result["acc_bag"] = sample["acc_bag"]
        if "traj_bag" in sample:
            result["traj_bag"] = sample["traj_bag"]
        return result


def collate_mil_pad(batch: List[Dict], *, concat_channels: bool = False):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    modality_keys = [key for key in ("acc_bag", "traj_bag") if key in batch[0]]
    if not modality_keys:
        raise KeyError("Batch does not contain acc_bag or traj_bag.")

    counts = [item[modality_keys[0]].shape[0] for item in batch]
    max_instances = max(counts)
    masks = torch.zeros(len(batch), max_instances, dtype=torch.bool)
    result = {
        "mask": masks,
        "y": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "y_task": torch.tensor(
            [item["task_target"] for item in batch], dtype=torch.long
        ),
        "meta": [item["meta"] for item in batch],
    }

    for batch_index, count in enumerate(counts):
        masks[batch_index, :count] = batch[batch_index]["mask"]

    for key in modality_keys:
        tensors = [item[key] for item in batch]
        channels = tensors[0].shape[1]
        time_steps = tensors[0].shape[2]
        padded = torch.zeros(
            len(batch),
            max_instances,
            channels,
            time_steps,
            dtype=tensors[0].dtype,
        )
        for batch_index, tensor in enumerate(tensors):
            padded[batch_index, :tensor.shape[0]] = tensor
        result[key] = padded

    if concat_channels:
        if set(modality_keys) != {"acc_bag", "traj_bag"}:
            raise ValueError("concat_channels=True requires multimodal data.")
        result["x_bag"] = torch.cat(
            [result.pop("acc_bag"), result.pop("traj_bag")],
            dim=2,
        )

    return result


def build_dataloader(
    manifest,
    val_pid,
    target_per_class=200,
    filter_task=None,
    batch_size=16,
    *,
    modality: str = "multimodal",
    segment_manifest: Optional[pd.DataFrame] = None,
    seg_len: int = SEG_LEN,
    hop: int = HOP,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
):
    set_seed()
    modality = _normalize_modality(modality)

    if segment_manifest is None:
        segment_manifest = build_segment_manifest(
            manifest,
            seg_len=seg_len,
            hop=hop,
            usecols=_columns_for_modality(modality),
            require_exists=True,
            keep_tail=False,
        )

    train_segments, valid_segments = split_train_valid_by_patient(
        segment_manifest,
        val_pid=[val_pid],
        reset_index=True,
    )
    bag_rot = BagRotator(seed=42, prob=1.0)
    inst_ptw = InstancePermTimeW(
        seed=42,
        n_min=1,
        n_max=5,
        tw_a=(0.15, 0.35),
        tw_f=(0.5, 1.5),
        apply_perm=True,
        apply_timew=True,
        prob_perm=1.0,
        prob_timew=1.0,
    )
    train_dataset = BalancedMILDatasetPath_RPT(
        train_segments,
        bag_rot=bag_rot,
        inst_perm_timew=inst_ptw,
        max_instances_per_bag=10,
        target_per_class=target_per_class,
        allow_downsample=False,
        print_stats=True,
        filter_task=filter_task,
        modality=modality,
    )
    valid_dataset = EvalMILDatasetPath(
        valid_segments,
        target_col="target",
        filter_task=filter_task,
        modality=modality,
    )

    worker_count = NUM_WORKERS if num_workers is None else int(num_workers)
    keep_workers = False if persistent_workers is None else persistent_workers
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": worker_count,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(keep_workers and worker_count > 0),
        "collate_fn": partial(collate_mil_pad, concat_channels=False),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader


def build_LOSO_loaders(
    manifest,
    target_per_class=200,
    filter_task=None,
    *,
    modality: str = "multimodal",
    patient_ids: Optional[Sequence[int]] = None,
    batch_size: int = 16,
    seg_len: int = SEG_LEN,
    hop: int = HOP,
    num_workers: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
):
    modality = _normalize_modality(modality)
    segment_manifest = build_segment_manifest(
        manifest,
        seg_len=seg_len,
        hop=hop,
        usecols=_columns_for_modality(modality),
        require_exists=True,
        keep_tail=False,
    )

    available_patient_ids = sorted(
        int(patient_id)
        for patient_id in pd.Series(manifest["patient_id"]).dropna().unique()
    )
    selected_patient_ids = (
        available_patient_ids
        if patient_ids is None
        else [int(patient_id) for patient_id in patient_ids]
    )
    unknown = sorted(set(selected_patient_ids) - set(available_patient_ids))
    if unknown:
        raise ValueError(f"Unknown patient_ids: {unknown}")

    loaders = {}
    for patient_id in selected_patient_ids:
        loaders[str(patient_id)] = build_dataloader(
            manifest,
            patient_id,
            target_per_class=target_per_class,
            filter_task=filter_task,
            batch_size=batch_size,
            modality=modality,
            segment_manifest=segment_manifest,
            seg_len=seg_len,
            hop=hop,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
    return loaders
