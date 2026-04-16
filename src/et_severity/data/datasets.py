"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *
from ..config import DEFAULT_USECOLS, HOP, NUM_WORKERS, SEG_LEN
from ..utils.reproducibility import set_seed
from .augmentations import BagRotator, InstancePermTimeW
from .preprocessing import expand_train_valid_to_segments, split_train_valid_by_patient

class _BaseMILLoaderPath:
    """
    seg_df: [path,start,end,target,task,task_idx] (+메타)
    bag 단위: path (CSV 하나가 한 bag)
    """
    def __init__(self, seg_df: pd.DataFrame, target_col: str = "target",
                 usecols: Optional[List[str]] = None, traj_div: float = 400.0,
                 filter_task: Optional[str] = None): # ▼▼▼ NEW/MODIFIED ▼▼▼: task 필터링 옵션 추가
        need = {"path","start","end",target_col,"task","task_idx"}
        miss = need - set(seg_df.columns)
        if miss: raise KeyError(f"필수 컬럼 누락: {miss}")

        self.df = seg_df.reset_index(drop=True).copy()
        self.target_col = target_col
        self.usecols = list(DEFAULT_USECOLS if usecols is None else usecols)
        self.traj_div = float(traj_div)
        self.filter_task = filter_task # ▼▼▼ NEW/MODIFIED ▼▼▼: 옵션 저장

        # ▼▼▼ NEW/MODIFIED ▼▼▼: 지정된 task가 있으면 데이터프레임 필터링
        if self.filter_task:
            self.df = self.df[self.df["task"] == self.filter_task].copy()
            print(f"[INFO] Dataset filtered for task: '{self.filter_task}'. Total segments: {len(self.df)}")
        # ▲▲▲ NEW/MODIFIED ▲▲▲

        self.df["_bagkey"] = self.df["path"].astype(str)
        self.groups = self.df.groupby("_bagkey")
        self.bags = []
        for k, g in self.groups:
            bag_y = int(g[self.target_col].mode().iloc[0])
            bag_task_idx = int(g["task_idx"].mode().iloc[0])
            bag_task_str = str(g["task"].mode().iloc[0])
            self.bags.append({
                "bagkey": k,
                "rows": g.sort_values("start"),
                "target": bag_y,
                "task_idx": bag_task_idx,
                "task": bag_task_str,
            })

    def load_bag(self, bag_idx: int):
        rows = self.bags[bag_idx]["rows"]
        acc_list, traj_list = [], []
        for _, r in rows.iterrows():
            d = pd.read_csv(r["path"], usecols=self.usecols)
            seg = d.iloc[int(r["start"]):int(r["end"])]
            acc  = seg[["accel_x","accel_y","accel_z"]].to_numpy(float).T
            traj = seg[["coor_x","coor_y"]].to_numpy(float).T / self.traj_div
            acc_list.append(torch.from_numpy(acc).float())
            traj_list.append(torch.from_numpy(traj).float())
        acc_bag  = torch.stack(acc_list, dim=0)
        traj_bag = torch.stack(traj_list, dim=0)
        return (
            acc_bag,
            traj_bag,
            int(self.bags[bag_idx]["target"]),
            int(self.bags[bag_idx]["task_idx"]),
            str(self.bags[bag_idx]["task"]),
            str(self.bags[bag_idx]["bagkey"]),
        )

class BalancedMILDatasetPath_RPT(Dataset):
    """
    - 원본 bag(aug=False)은 그대로 유지 (allow_downsample=False일 때)
    - 클래스 비율 균등 유지
    - target_per_class 또는 target_total_bags로 최종 bag 수를 제어(원본 보존 제약 하에서)
    """
    def __init__(self, seg_train_df: pd.DataFrame,
                 target_col: str = "target",
                 bag_rot: Optional[BagRotator] = None,
                 inst_perm_timew: Optional[InstancePermTimeW] = None,
                 max_instances_per_bag: Optional[int] = None,
                 print_stats: bool = True,
                 target_per_class: Optional[int] = None,
                 target_total_bags: Optional[int] = None,
                 allow_downsample: bool = False,
                 seed_balance: int = 2024,
                 filter_task: Optional[str] = None): # ▼▼▼ NEW/MODIFIED ▼▼▼: task 필터링 옵션 추가
        # ▼▼▼ NEW/MODIFIED ▼▼▼: _BaseMILLoaderPath에 filter_task 전달
        self.base = _BaseMILLoaderPath(seg_train_df, target_col=target_col, filter_task=filter_task)
        self.bag_rot = bag_rot
        self.inst_ptw = inst_perm_timew
        self.maxN = max_instances_per_bag

        self.orig_indices = list(range(len(self.base.bags)))
        targets = [b["target"] for b in self.base.bags]
        counter = collections.Counter(targets)
        self.classes = sorted(counter.keys())
        self.orig_counts = {c: counter[c] for c in self.classes}
        K = len(self.classes)
        self.max_count = max(counter.values()) if counter else 0

        # --- 목표 개수 계산 (클래스 균등) ---
        if target_per_class is not None:
            desired_per_class = {c: int(target_per_class) for c in self.classes}
        elif target_total_bags is not None and K > 0:
            base = target_total_bags // K
            rem  = target_total_bags % K
            desired_per_class = {c: base + (i < rem) for i, c in enumerate(self.classes)}
        else:
            desired_per_class = {c: self.max_count for c in self.classes}

        if not allow_downsample:
            for c in self.classes:
                if desired_per_class[c] < self.orig_counts[c]:
                    desired_per_class[c] = self.orig_counts[c]

        # --- 아이템 구성: 원본(aug=False) + 증강복제(aug=True) ---
        per_class_indices = {c: [i for i, b in enumerate(self.base.bags) if b["target"] == c] for c in self.classes}
        aug_items = []
        rng = np.random.default_rng(seed_balance)

        kept_original_indices = set(self.orig_indices)
        if allow_downsample:
            new_kept = set()
            for c in self.classes:
                cls_indices = per_class_indices[c]
                keep_n = min(len(cls_indices), desired_per_class[c])
                if keep_n < len(cls_indices):
                    chosen = rng.choice(cls_indices, size=keep_n, replace=False).tolist()
                else:
                    chosen = cls_indices
                new_kept.update(chosen)
            kept_original_indices = new_kept

        for c in self.classes:
            cls_indices = per_class_indices[c]
            kept_n = len([i for i in cls_indices if i in kept_original_indices])
            need = desired_per_class[c] - kept_n
            if need > 0 and kept_n > 0:
                pool = [i for i in cls_indices if i in kept_original_indices]
                sampled = rng.integers(0, len(pool), size=need)
                for s in sampled:
                    aug_items.append((pool[int(s)], True))

        self.items = [(i, False) for i in sorted(kept_original_indices)] + aug_items

        if print_stats and len(self.items) > 0:
            before = dict(self.orig_counts)
            after_counts = collections.Counter([self.base.bags[i]["target"] for (i, is_aug) in self.items if not is_aug])
            total_counts = {c: after_counts.get(c, 0) for c in self.classes}
            for (bi, _) in aug_items:
                c = self.base.bags[bi]["target"]
                total_counts[c] = total_counts.get(c, 0) + 1

            mism = {c: (total_counts[c], desired_per_class[c]) for c in self.classes if total_counts[c] != desired_per_class[c]}
            if mism:
                print("[WARN] Desired per-class counts not exactly met (likely due to original-count constraints):", mism)

            print("[MIL Balance (R+P+T) @ path-bag]")
            print(f"  Task Filter: {self.base.filter_task or 'None'}")
            print("  before :", before)
            print("  desired:", desired_per_class)
            print("  after  :", total_counts, "| total:", len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        bag_idx, is_aug = self.items[idx]
        acc_bag, traj_bag, y, y_task, task_str, bagkey = self.base.load_bag(bag_idx)

        if is_aug:
            if self.bag_rot is not None:
                acc_bag, traj_bag = self.bag_rot(acc_bag, traj_bag)
            if self.inst_ptw is not None:
                accs, trajs = [], []
                for j in range(acc_bag.shape[0]):
                    a = acc_bag[j].cpu().numpy()
                    t = traj_bag[j].cpu().numpy()
                    a2, t2 = self.inst_ptw(a, t)
                    accs.append(torch.from_numpy(a2).float())
                    trajs.append(torch.from_numpy(t2).float())
                acc_bag = torch.stack(accs, dim=0)
                traj_bag = torch.stack(trajs, dim=0)

        if (self.maxN is not None) and (acc_bag.shape[0] > self.maxN):
            N = acc_bag.shape[0]
            idx_keep = torch.randperm(N)[:self.maxN]
            acc_bag = acc_bag[idx_keep]
            traj_bag = traj_bag[idx_keep]

        mask = torch.ones(acc_bag.shape[0], dtype=torch.bool)
        meta = {"bagkey": bagkey, "N": acc_bag.shape[0], "aug": bool(is_aug), "task": task_str}
        return {
            "acc_bag": acc_bag,
            "traj_bag": traj_bag,
            "mask": mask,
            "target": y,
            "task_target": y_task,
            "meta": meta
        }

class EvalMILDatasetPath(Dataset):
    def __init__(self, seg_valid_df: pd.DataFrame, target_col: str="target",
                 filter_task: Optional[str] = None): # ▼▼▼ NEW/MODIFIED ▼▼▼: task 필터링 옵션 추가
        # ▼▼▼ NEW/MODIFIED ▼▼▼: _BaseMILLoaderPath에 filter_task 전달
        self.base = _BaseMILLoaderPath(seg_valid_df, target_col=target_col, filter_task=filter_task)

    def __len__(self):
        return len(self.base.bags)

    def __getitem__(self, i):
        acc_bag, traj_bag, y, y_task, task_str, bagkey = self.base.load_bag(i)
        mask = torch.ones(acc_bag.shape[0], dtype=torch.bool)
        return {
            "acc_bag": acc_bag,
            "traj_bag": traj_bag,
            "mask": mask,
            "target": y,
            "task_target": y_task,
            "meta": {"bagkey": bagkey, "N": acc_bag.shape[0], "aug": False, "task": task_str}
        }

def collate_mil_pad(batch: List[Dict], *, concat_channels: bool = True):
    B = len(batch)
    Ns = [b["acc_bag"].shape[0] for b in batch]
    Cacc = batch[0]["acc_bag"].shape[1]
    Ctraj= batch[0]["traj_bag"].shape[1]
    T    = batch[0]["acc_bag"].shape[2]
    Nmax = max(Ns) if Ns else 0

    def pad_stack(tensors, C):
        out = torch.zeros(B, Nmax, C, T, dtype=tensors[0].dtype)
        for i, x in enumerate(tensors):
            out[i, :x.shape[0]] = x
        return out

    accs  = [b["acc_bag"]  for b in batch]
    trajs = [b["traj_bag"] for b in batch]
    masks = torch.zeros(B, Nmax, dtype=torch.bool)
    ys    = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    ys_task = torch.tensor([b["task_target"] for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        masks[i, :Ns[i]] = b["mask"]

    acc_pad  = pad_stack(accs,  Cacc)
    traj_pad = pad_stack(trajs, Ctraj)

    if concat_channels:
        x = torch.cat([acc_pad, traj_pad], dim=2)
        return {
            'x_bag':x,
            'mask':masks,
            'y':ys,
            'y_task':ys_task,
            'meta':[b["meta"] for b in batch]
        }
    else:
        return {
            'acc_bag':acc_pad,
            'traj_bag':traj_pad,
            'mask':masks,
            'y':ys,
            'y_task':ys_task,
            'meta':[b["meta"] for b in batch]
        }

def build_dataloader(
    manifest,
    val_pid,
    target_per_class=200,
    filter_task=None,
    batch_size=16,
    *,
    seg_len: int = SEG_LEN,
    hop: int = HOP,
):
    set_seed()

    tr_man, val_man = split_train_valid_by_patient(
        manifest,
        val_pid=[val_pid],
        reset_index=True
    )
    tr_segman, val_segman = expand_train_valid_to_segments(
        tr_man, val_man,
        seg_len=seg_len,
        hop=hop,
        require_exists=True,
        keep_tail=False,
    )
    bag_rot = BagRotator(seed=42, prob=1.0)
    inst_ptw = InstancePermTimeW(seed=42, n_min=1, n_max=5,
                                tw_a=(0.15,0.35), tw_f=(0.5,1.5),
                                apply_perm=True, apply_timew=True,
                                prob_perm=1.0, prob_timew=1.0)


    # ② 클래스당 개수로 직접 지정(예: 각 클래스 180개)
    mil_train = BalancedMILDatasetPath_RPT(
        tr_segman,
        bag_rot=bag_rot, inst_perm_timew=inst_ptw,
        max_instances_per_bag=10,
        target_per_class=target_per_class,       # ← 추가
        allow_downsample=False,
        print_stats=True,
        filter_task=filter_task
    )

    mil_valid = EvalMILDatasetPath(val_segman, target_col="target", filter_task=filter_task)

    pin_memory = torch.cuda.is_available()
    dl_tr = DataLoader(mil_train, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS,
                    pin_memory=pin_memory, collate_fn=lambda b: collate_mil_pad(b, concat_channels=False))
    dl_va = DataLoader(mil_valid,     batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                    pin_memory=pin_memory, collate_fn=lambda b: collate_mil_pad(b, concat_channels=False))
    return dl_tr, dl_va

def build_LOSO_loaders(
    manifest,
    target_per_class=200,
    filter_task=None,
    *,
    batch_size: int = 16,
    seg_len: int = SEG_LEN,
    hop: int = HOP,
):
    loaders = {}
    patient_ids = sorted(int(pid) for pid in pd.Series(manifest["patient_id"]).dropna().unique())
    for patient_id in patient_ids:
        dl_tr, dl_va = build_dataloader(
            manifest,
            patient_id,
            target_per_class=target_per_class,
            filter_task=filter_task,
            batch_size=batch_size,
            seg_len=seg_len,
            hop=hop,
        )
        loaders[str(patient_id)] = (dl_tr, dl_va)
    return loaders


# Cleaner public aliases for import-based notebooks and scripts.
PathBagLoaderBase = _BaseMILLoaderPath
BalancedBagDataset = BalancedMILDatasetPath_RPT
EvaluationBagDataset = EvalMILDatasetPath
pad_bag_batch = collate_mil_pad
build_patient_holdout_loaders = build_dataloader
build_loso_loaders = build_LOSO_loaders
