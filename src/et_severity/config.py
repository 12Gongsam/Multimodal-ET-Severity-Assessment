"""Shared configuration defaults for package modules and notebooks."""

from __future__ import annotations

import re

import torch

SEED = 42
FS = 100.0
SEG_LEN = 512
HOP = 512

DEFAULT_D_MODEL = 128
DEFAULT_MIL_ATTN_DIM = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_TARGET_PER_CLASS = 200
DEFAULT_USECOLS = ["accel_x", "accel_y", "accel_z", "coor_x", "coor_y"]

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if DEFAULT_DEVICE.type == "cuda" else 2

DIR_RE = re.compile(r"ET(\d{2})_(\d)_(\d{8})")
TASK_RE = re.compile(r"(Spiral|Maze|MultiDirect)", re.IGNORECASE)
TASK2IDX = {"spiral": 0, "maze": 1, "multidirect": 2, "unknown": 3}
SPLIT_FILE_RE = re.compile(r".*_\d+_[123]$")
