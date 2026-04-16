"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *

def _compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Args:
        logits: [B, C]
        labels: [B]
    Returns:
        dict with:
         - acc, correct, total
         - conf: [C,C] confusion matrix (rows=GT, cols=Pred)
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        acc = correct / max(1, total)

        C = logits.size(1)
        # rows = GT, cols = Pred
        idx = (labels.long() * C + preds.long()).view(-1)
        conf = torch.bincount(idx, minlength=C * C).reshape(C, C)

    return {"acc": acc, "correct": float(correct), "total": float(total), "conf": conf}

def _confmat_to_perclass_acc_macro_f1(conf: torch.Tensor) -> Dict[str, object]:
    """
    # ▼▼▼ MODIFIED ▼▼▼
    # Tensors are now created on the same device as the input 'conf' tensor to prevent runtime errors.
    # GT(Ground Truth)에 등장한 클래스에 대해서만 macro_f1을 계산하고,
    # 등장하지 않은 클래스의 per-class acc는 NaN으로 표시합니다.
    # ▲▲▲ MODIFIED ▲▲▲
    """
    conf = conf.float()
    C = conf.size(0)
    if C == 0:
        return {"per_class_acc": [], "macro_f1": 0.0}

    device = conf.device  # Get device from input tensor
    support = conf.sum(dim=1)      # GT count per class
    pred_cnt = conf.sum(dim=0)     # Pred count per class
    tp = conf.diag()

    # per-class accuracy (= recall per class)
    # Create tensor on the correct device
    per_class_acc = torch.full((C,), float('nan'), dtype=torch.float32, device=device)
    mask_sup = support > 0
    per_class_acc[mask_sup] = tp[mask_sup] / support[mask_sup]

    # precision, recall, f1 per class
    # Create tensors on the correct device
    precision = torch.zeros(C, dtype=torch.float32, device=device)
    recall    = torch.zeros(C, dtype=torch.float32, device=device)
    precision[pred_cnt > 0] = tp[pred_cnt > 0] / pred_cnt[pred_cnt > 0]
    recall[mask_sup]        = tp[mask_sup] / support[mask_sup]

    denom = precision + recall
    f1 = torch.zeros(C, dtype=torch.float32, device=device)
    nz = denom > 0
    f1[nz] = 2 * precision[nz] * recall[nz] / denom[nz]

    # GT에 등장한 클래스(support > 0)에 대해서만 F1 평균 계산
    macro_f1 = float(f1[mask_sup].mean().item()) if mask_sup.any() else 0.0

    return {
        "per_class_acc": per_class_acc.cpu().tolist(),
        "macro_f1": macro_f1,
    }

def evaluate_on_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    target: str = "severity",
    modality: Optional[str] = None
) -> Dict[str, object]:
    """
    선택된 타깃/모달리티로 평가합니다.
    (내부 로직은 변경 없음, _confmat_to_perclass_acc_macro_f1와 로직 일관성 확인)
    """
    assert target in ("severity", "task"), "target must be 'severity' or 'task'"
    assert modality in (None, "acc", "traj"), "modality must be None, 'acc', or 'traj'"

    # 내부 메트릭 계산 함수. _confmat_to_perclass_acc_macro_f1와 동일한 로직 사용.
    def _metrics_from_conf(conf: torch.Tensor):
        conf = conf.to(torch.float32)
        C = conf.size(0)
        support = conf.sum(dim=1)
        pred_cnt = conf.sum(dim=0)
        tp = torch.diag(conf)

        per_class_acc = torch.full((C,), float('nan'), dtype=torch.float32)
        mask_sup = support > 0
        per_class_acc[mask_sup] = tp[mask_sup] / support[mask_sup]

        precision = torch.zeros(C, dtype=torch.float32)
        recall    = torch.zeros(C, dtype=torch.float32)
        precision[pred_cnt > 0] = tp[pred_cnt > 0] / pred_cnt[pred_cnt > 0]
        recall[mask_sup]        = tp[mask_sup] / support[mask_sup]

        denom = precision + recall
        f1 = torch.zeros(C, dtype=torch.float32)
        nz = denom > 0
        f1[nz] = 2 * precision[nz] * recall[nz] / denom[nz]

        macro_f1 = float(f1[mask_sup].mean().item()) if mask_sup.any() else 0.0
        return {"macro_f1": macro_f1, "per_class_acc": per_class_acc.tolist()}

    model.eval()
    epoch_conf = None
    correct = 0.0
    total = 0.0

    pbar = tqdm(loader, desc=f"Eval({target}|{modality or 'both'})", leave=False)
    for batch in pbar:
        acc  = batch["acc_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)
        traj = batch["traj_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y = (batch["y"] if target == "severity" else batch["y_task"]).to(device, non_blocking=True)

        if modality is None:
            out = model(acc, traj, mask=mask)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                sev_logits, task_logits = out[0], out[1]
                logits = sev_logits if target == "severity" else task_logits
            else:
                logits = out if torch.is_tensor(out) else out[0]
        elif modality == "acc":
            out = model(acc, mask=mask)
            logits = out[0] if (isinstance(out, (tuple, list)) and len(out) > 0) else out
        else:
            out = model(traj, mask=mask)
            logits = out[0] if (isinstance(out, (tuple, list)) and len(out) > 0) else out

        preds = logits.argmax(dim=-1)
        y_cpu = y.detach().cpu()
        p_cpu = preds.detach().cpu()

        C = logits.size(-1)
        if epoch_conf is None:
            epoch_conf = torch.zeros(C, C, dtype=torch.long)

        for c in range(C):
            m = (y_cpu == c)
            if m.any():
                binc = torch.bincount(p_cpu[m], minlength=C)
                epoch_conf[c, :] += binc

        correct += (p_cpu == y_cpu).sum().item()
        total   += float(y_cpu.numel())

        mm_now = _metrics_from_conf(epoch_conf)
        acc_now = correct / max(1.0, total)
        pbar.set_postfix({"acc": f"{acc_now:.3f}", "macro_f1": f"{mm_now['macro_f1']:.3f}"})

    if epoch_conf is None:
        return {"acc": 0.0, "macro_f1": 0.0, "per_class_acc": [], "conf": torch.zeros(0, 0)}

    acc_final = correct / max(1.0, total)
    mm = _metrics_from_conf(epoch_conf)
    return {
        "acc": float(acc_final),
        "macro_f1": float(mm["macro_f1"]),
        "per_class_acc": mm["per_class_acc"],
        "conf": epoch_conf,
    }

def _run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = 1.0,
    lamda: float = 0.3,
    task_class_weights: Optional[torch.Tensor] = None # 'tw' -> 'task_class_weights'로 명확화
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    desc = "Train" if is_train else "Valid"

    tot_loss, tot_sev_loss, tot_task_loss = 0.0, 0.0, 0.0
    n_batches = 0
    sev_correct, sev_total = 0.0, 0.0
    task_correct, task_total = 0.0, 0.0
    epoch_conf_sev, epoch_conf_task = None, None

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        acc  = batch["acc_bag"].permute(0, 1, 3, 2).to(device)
        traj = batch["traj_bag"].permute(0, 1, 3, 2).to(device)
        mask = batch["mask"].to(device)
        y    = batch["y"].to(device)
        y_t  = batch["y_task"].to(device)

        with torch.set_grad_enabled(is_train):
            sev_logits, task_logits, _ = model(acc, traj, mask=mask)
            sev_loss = criterion(sev_logits, y)

            task_loss = F.cross_entropy(task_logits, y_t, weight=task_class_weights)

            loss = sev_loss + lamda * task_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        sm = _compute_metrics(sev_logits.detach(), y.detach())
        tm = _compute_metrics(task_logits.detach(), y_t.detach())

        epoch_conf_sev = sm["conf"].clone() if epoch_conf_sev is None else epoch_conf_sev + sm["conf"]
        epoch_conf_task = tm["conf"].clone() if epoch_conf_task is None else epoch_conf_task + tm["conf"]

        sev_correct += sm["correct"]; sev_total += sm["total"]
        task_correct += tm["correct"]; task_total += tm["total"]

        tot_loss += float(loss.detach().item())
        tot_sev_loss += float(sev_loss.detach().item())
        tot_task_loss += float(task_loss.detach().item())
        n_batches += 1

        sev_mm = _confmat_to_perclass_acc_macro_f1(epoch_conf_sev)
        task_mm = _confmat_to_perclass_acc_macro_f1(epoch_conf_task)

        pbar.set_postfix({
            "sev_loss": f"{tot_sev_loss / n_batches:.4f}",
            "task_loss": f"{tot_task_loss / n_batches:.4f}",
            "loss": f"{tot_loss / n_batches:.4f}",
            "sev_acc": f"{(sev_correct / max(1.0, sev_total)):.3f}",
            "task_acc": f"{(task_correct / max(1.0, task_total)):.3f}",
            "sev_f1": f"{sev_mm['macro_f1']:.3f}",
            "task_f1": f"{task_mm['macro_f1']:.3f}",
        })

    sev_mm = _confmat_to_perclass_acc_macro_f1(epoch_conf_sev if epoch_conf_sev is not None else torch.zeros(0,0))
    task_mm = _confmat_to_perclass_acc_macro_f1(epoch_conf_task if epoch_conf_task is not None else torch.zeros(0,0))

    return {
        "loss": tot_loss / max(1, n_batches),
        "sev_loss": tot_sev_loss / max(1, n_batches),
        "task_loss": tot_task_loss / max(1, n_batches),
        "sev_acc": sev_correct / max(1.0, sev_total),
        "task_acc": task_correct / max(1.0, task_total),
        "sev_macro_f1": sev_mm["macro_f1"],
        "task_macro_f1": task_mm["macro_f1"],
        "sev_cls_acc": sev_mm["per_class_acc"],
        "task_cls_acc": task_mm["per_class_acc"],
    }

def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    lamda: float = 0.3,
    task_class_weights: Optional[torch.Tensor] = None, # 'tw' -> 'task_class_weights'로 명확화
    use_scheduler: bool = True,
    sched_mode: str = "min",
    sched_factor: float = 0.5,
    sched_patience: int = 3,
    sched_cooldown: int = 1,
    sched_min_lr: float = 1e-6,
    sched_threshold: float = 1e-3,
    sched_threshold_mode: str = "rel",
    sched_eps: float = 1e-8,
) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, factor=sched_factor, patience=sched_patience,
            threshold=sched_threshold, threshold_mode=sched_threshold_mode,
            cooldown=sched_cooldown, min_lr=sched_min_lr, eps=sched_eps,
        )

    def _get_lr(opt): return opt.param_groups[0]["lr"]

    best_val_metric = float("inf") if sched_mode == "min" else float("-inf")
    best_state = None
    patience_counter = 0

    history = {k: [] for k in [
        "train_loss", "train_sev_loss", "train_task_loss", "train_sev_acc", "train_task_acc",
        "valid_loss", "valid_sev_loss", "valid_task_loss", "valid_sev_acc", "valid_task_acc",
        "train_sev_macro_f1", "train_task_macro_f1",
        "valid_sev_macro_f1", "valid_task_macro_f1",
        "train_sev_cls_acc", "train_task_cls_acc",
        "valid_sev_cls_acc", "valid_task_cls_acc", "lr",
    ]}

    for ep in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, device, criterion, optimizer=optimizer, grad_clip=grad_clip, lamda=lamda, task_class_weights=task_class_weights)
        va = _run_epoch(model, valid_loader, device, criterion, optimizer=None, grad_clip=None, lamda=lamda, task_class_weights=task_class_weights)

        cur_lr = _get_lr(optimizer)
        if scheduler is not None:
            monitor_metric = va["sev_loss"] if sched_mode == "min" else va["sev_macro_f1"]
            scheduler.step(monitor_metric)

        # Record history
        for k, v in tr.items(): history[f"train_{k}"].append(v)
        for k, v in va.items(): history[f"valid_{k}"].append(v)
        history["lr"].append(cur_lr)

        # ▼▼▼ NEW/MODIFIED ▼▼▼: NaN 값을 포함할 수 있는 per-class accuracy 배열을 안전하게 출력
        def fmt_arr(a):
            return "[" + ", ".join(f"{'nan' if np.isnan(x) else f'{x:.3f}'}" for x in a) + "]"
        # ▲▲▲ NEW/MODIFIED ▲▲▲

        print(
            f"[Epoch {ep:03d}] "
            f"TR loss {tr['loss']:.4f} (sev {tr['sev_loss']:.4f}, task {tr['task_loss']:.4f}) | "
            f"VA loss {va['loss']:.4f} (sev {va['sev_loss']:.4f}, task {va['task_loss']:.4f}) | "
            f"lr {history['lr'][-1]:.2e}\n"
            f"       TR acc (sev {tr['sev_acc']:.3f}, task {tr['task_acc']:.3f}) | "
            f"VA acc (sev {va['sev_acc']:.3f}, task {va['task_acc']:.3f})\n"
            f"       TR F1 (sev {tr['sev_macro_f1']:.3f}, task {tr['task_macro_f1']:.3f}) | "
            f"VA F1 (sev {va['sev_macro_f1']:.3f}, task {va['task_macro_f1']:.3f})\n"
            f"       TR per-class acc (sev {fmt_arr(tr['sev_cls_acc'])}, task {fmt_arr(tr['task_cls_acc'])})\n"
            f"       VA per-class acc (sev {fmt_arr(va['sev_cls_acc'])}, task {fmt_arr(va['task_cls_acc'])})"
        )
        print()

        # Early stopping check
        current_metric = va["sev_loss"] if sched_mode == "min" else va["sev_macro_f1"]
        is_better = (current_metric < best_val_metric) if sched_mode == "min" else (current_metric > best_val_metric)

        if is_better:
            best_val_metric = current_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {ep} (no improvement for {early_stopping_patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

def _unpack_sev_only(model_out):
    if isinstance(model_out, (tuple, list)):
        if len(model_out) == 2:
            sev_logits, aux = model_out
            return sev_logits, aux
        elif len(model_out) == 3:
            sev_logits, _task_logits, aux = model_out
            return sev_logits, aux
    # 모델이 sev_logits만 주는 경우
    return model_out, {}

def _run_epoch_sev_only(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    sev(중증도) 단일 과제 학습/검증 루프.
    - task 로짓/로스/메트릭 일체 제거
    - confmat 기반 macro-F1 및 per-class acc 계산 (_compute_metrics, _confmat_to_perclass_acc_macro_f1 사용)
    """
    is_train = optimizer is not None
    model.train(mode=is_train)
    desc = "Train" if is_train else "Valid"

    tot_loss = 0.0
    n_batches = 0
    sev_correct = 0.0
    sev_total = 0.0
    epoch_conf_sev = None

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        acc  = batch["acc_bag"].permute(0, 1, 3, 2).to(device)
        traj = batch["traj_bag"].permute(0, 1, 3, 2).to(device)
        mask = batch["mask"].to(device)
        y    = batch["y"].to(device)

        with torch.set_grad_enabled(is_train):
            sev_logits, _ = _unpack_sev_only(model(acc, traj, mask=mask))
            sev_loss = criterion(sev_logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                sev_loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        sm = _compute_metrics(sev_logits.detach(), y.detach())
        epoch_conf_sev = sm["conf"].clone() if epoch_conf_sev is None else epoch_conf_sev + sm["conf"]

        sev_correct += sm["correct"]
        sev_total   += sm["total"]
        tot_loss += float(sev_loss.detach().item())
        n_batches += 1

        sev_mm = _confmat_to_perclass_acc_macro_f1(epoch_conf_sev)
        pbar.set_postfix({
            "sev_loss": f"{tot_loss / n_batches:.4f}",
            "sev_acc":  f"{(sev_correct / max(1.0, sev_total)):.3f}",
            "sev_f1":   f"{sev_mm['macro_f1']:.3f}",
        })

    sev_mm = _confmat_to_perclass_acc_macro_f1(epoch_conf_sev if epoch_conf_sev is not None else torch.zeros(0,0))
    return {
        "loss":         tot_loss / max(1, n_batches),
        "sev_loss":     tot_loss / max(1, n_batches),
        "sev_acc":      sev_correct / max(1.0, sev_total),
        "sev_macro_f1": sev_mm["macro_f1"],
        "sev_cls_acc":  sev_mm["per_class_acc"],
    }

def train_model_sev_only(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    device: torch.device,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    use_scheduler: bool = True,
    sched_mode: str = "min",
    sched_factor: float = 0.5,
    sched_patience: int = 3,
    sched_cooldown: int = 1,
    sched_min_lr: float = 1e-6,
    sched_threshold: float = 1e-3,
    sched_threshold_mode: str = "rel",
    sched_eps: float = 1e-8,
) -> Dict[str, List[float]]:
    """
    Severity 단일 태스크 학습 루프.
    """
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, factor=sched_factor, patience=sched_patience,
            threshold=sched_threshold, threshold_mode=sched_threshold_mode,
            cooldown=sched_cooldown, min_lr=sched_min_lr, eps=sched_eps,
        )

    def _get_lr(opt): return opt.param_groups[0]["lr"]

    best_val_metric = float("inf") if sched_mode == "min" else float("-inf")
    best_state = None
    patience_counter = 0

    history = {k: [] for k in [
        "train_loss", "train_sev_loss", "train_sev_acc", "train_sev_macro_f1", "train_sev_cls_acc",
        "valid_loss", "valid_sev_loss", "valid_sev_acc", "valid_sev_macro_f1", "valid_sev_cls_acc",
        "lr",
    ]}

    for ep in range(1, epochs + 1):
        tr = _run_epoch_sev_only(model, train_loader, device, criterion, optimizer=optimizer, grad_clip=grad_clip)
        va = _run_epoch_sev_only(model, valid_loader, device, criterion, optimizer=None, grad_clip=None)

        cur_lr = _get_lr(optimizer)
        if scheduler is not None:
            monitor_metric = va["sev_loss"] if sched_mode == "min" else va["sev_macro_f1"]
            scheduler.step(monitor_metric)
        history["lr"].append(cur_lr)

        # 기록
        history_keys = list(tr.keys())
        for k in history_keys:
            if f"train_{k}" in history: history[f"train_{k}"].append(tr[k])
            if f"valid_{k}" in history: history[f"valid_{k}"].append(va[k])

        # ▼▼▼ NEW/MODIFIED ▼▼▼: NaN 값을 포함할 수 있는 per-class accuracy 배열을 안전하게 출력
        def fmt_arr(a):
            if not isinstance(a, list): return str(a)
            return "[" + ", ".join(f"{'nan' if np.isnan(x) else f'{x:.3f}'}" for x in a) + "]"
        # ▲▲▲ NEW/MODIFIED ▲▲▲

        print(
            f"[Epoch {ep:03d}] "
            f"TR loss {tr['loss']:.4f} | VA loss {va['loss']:.4f} | lr {history['lr'][-1]:.2e}\n"
            f"       TR acc/F1 (sev {tr['sev_acc']:.3f} / {tr['sev_macro_f1']:.3f}) | "
            f"VA acc/F1 (sev {va['sev_acc']:.3f} / {va['sev_macro_f1']:.3f})\n"
            f"       TR per-class acc (sev {fmt_arr(tr['sev_cls_acc'])})\n"
            f"       VA per-class acc (sev {fmt_arr(va['sev_cls_acc'])})"
        )
        print()

        current_metric = va["sev_loss"] if sched_mode == "min" else va["sev_macro_f1"]
        is_better = (current_metric < best_val_metric) if sched_mode == "min" else (current_metric > best_val_metric)

        if is_better:
            best_val_metric = current_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {ep} (no improvement for {early_stopping_patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

def evaluate_on_loader_sev_only( # ▼▼▼ NEW/MODIFIED ▼▼▼: 함수 이름 변경으로 충돌 방지
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    target: str = "severity",
    modality: Optional[str] = None
) -> Dict[str, object]:
    assert target == "severity", "This evaluation function only supports 'severity'."
    assert modality in (None, "acc", "traj"), "modality must be None, 'acc', or 'traj'"

    # 내부 메트릭 계산 함수는 이미 올바르게 구현되어 있음
    def _metrics_from_conf(conf: torch.Tensor):
        return _confmat_to_perclass_acc_macro_f1(conf)

    model.eval()
    epoch_conf = None
    correct = 0.0
    total = 0.0

    pbar = tqdm(loader, desc=f"Eval(severity|{modality or 'both'})", leave=False)
    for batch in pbar:
        acc  = batch["acc_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)
        traj = batch["traj_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        y    = batch["y"].to(device, non_blocking=True)

        if modality is None:
            out = model(acc, traj, mask=mask)
        elif modality == "acc":
            out = model(acc,  mask=mask)
        else: # "traj"
            out = model(traj, mask=mask)

        sev_logits, _ = _unpack_sev_only(out)

        preds = sev_logits.argmax(dim=-1)
        y_cpu = y.detach().cpu()
        p_cpu = preds.detach().cpu()

        C = sev_logits.size(-1)
        if epoch_conf is None:
            epoch_conf = torch.zeros(C, C, dtype=torch.long)

        for c in range(C):
            m = (y_cpu == c)
            if m.any():
                binc = torch.bincount(p_cpu[m], minlength=C)
                epoch_conf[c, :] += binc

        correct += (p_cpu == y_cpu).sum().item()
        total   += float(y_cpu.numel())

        mm_now = _metrics_from_conf(epoch_conf)
        acc_now = correct / max(1.0, total)
        pbar.set_postfix({"acc": f"{acc_now:.3f}", "macro_f1": f"{mm_now['macro_f1']:.3f}"})

    if epoch_conf is None:
        return {"acc": 0.0, "macro_f1": 0.0, "per_class_acc": [], "conf": torch.zeros(0, 0)}

    acc_final = correct / max(1.0, total)
    mm = _metrics_from_conf(epoch_conf)
    return {
        "acc": float(acc_final),
        "macro_f1": float(mm["macro_f1"]),
        "per_class_acc": mm["per_class_acc"],
        "conf": epoch_conf,
    }

def _run_epoch_single(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    is_train: bool,
    target: str,                    # "severity" or "task"
    modality: str,                  # "acc" or "traj"
    criterion: nn.Module = nn.CrossEntropyLoss(),
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    assert target in ("severity", "task")
    assert modality in ("acc", "traj")

    model.train(mode=is_train)

    tot_loss = 0.0
    n_batches = 0
    correct = 0.0
    total   = 0.0
    epoch_conf = None  # [C,C] (CPU)

    pbar = tqdm(loader, desc=("Train(single)" if is_train else "Valid(single)"), leave=False)
    for batch in pbar:
        # 입력/전처리
        if modality == "acc":
            x_bag = batch["acc_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)   # [B,N,T,C]
        else:
            x_bag = batch["traj_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)  # [B,N,T,C]

        mask = batch["mask"].to(device, non_blocking=True)  # [B,N] (bool)
        y = (batch["y"] if target == "severity" else batch["y_task"]).to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits, _, _ = model(x_bag, mask=mask)  # [B,C]

            if class_weights is not None:
                loss = F.cross_entropy(logits, y, weight=class_weights.to(device=device, dtype=torch.float))
            else:
                loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        # Metric 누적
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total   += float(y.numel())

            C = logits.size(-1)
            y_cpu = y.detach().cpu()
            p_cpu = preds.detach().cpu()

            if epoch_conf is None:
                epoch_conf = torch.zeros(C, C, dtype=torch.long)

            for c in range(C):
                m = (y_cpu == c)
                if m.any():
                    binc = torch.bincount(p_cpu[m], minlength=C)
                    epoch_conf[c, :] += binc

        tot_loss += float(loss.detach().item())
        n_batches += 1

        acc_now = correct / max(1.0, total)
        mm_now = _confmat_to_perclass_acc_macro_f1(epoch_conf)
        pbar.set_postfix({"loss": f"{tot_loss / n_batches:.4f}",
                          "acc": f"{acc_now:.3f}",
                          "macro_f1": f"{mm_now['macro_f1']:.3f}"})

    if epoch_conf is None:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0, "per_class_acc": [], "conf": torch.zeros(0, 0)}

    mm_final = _confmat_to_perclass_acc_macro_f1(epoch_conf)
    return {
        "loss": tot_loss / max(1, n_batches),
        "acc": correct / max(1.0, total),
        "macro_f1": mm_final["macro_f1"],
        "per_class_acc": mm_final["per_class_acc"],
        "conf": epoch_conf,
    }

def train_model_single(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    device: torch.device,
    modality: str,                 # "acc" or "traj"
    target: str,                   # "severity" or "task"
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    use_scheduler: bool = True,
    sched_mode: str = "min",       # "min": val loss, "max": val macro_f1
    sched_factor: float = 0.5,
    sched_patience: int = 3,
    sched_cooldown: int = 1,
    sched_min_lr: float = 1e-6,
    sched_threshold: float = 1e-3,
    sched_threshold_mode: str = "rel",
    sched_eps: float = 1e-8,
    class_weights: Optional[torch.Tensor] = None,   # size=num_classes
) -> Dict[str, List[float]]:
    assert modality in ("acc", "traj")
    assert target in ("severity", "task")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_mode,
            factor=sched_factor,
            patience=sched_patience,
            threshold=sched_threshold,
            threshold_mode=sched_threshold_mode,
            cooldown=sched_cooldown,
            min_lr=sched_min_lr,
            eps=sched_eps,
        )

    def _get_lr(opt):
        return opt.param_groups[0]["lr"]

    best_val_score = math.inf if sched_mode == "min" else -math.inf
    best_state = None
    patience = 0

    history = {
        "train_loss": [], "train_acc": [], "train_macro_f1": [], "train_cls_acc": [],
        "valid_loss": [], "valid_acc": [], "valid_macro_f1": [], "valid_cls_acc": [],
        "lr": [],
    }

    for ep in range(1, epochs + 1):
        tr = _run_epoch_single(
            model, train_loader, device,
            is_train=True, target=target, modality=modality,
            optimizer=optimizer, grad_clip=grad_clip, class_weights=class_weights
        )
        va = _run_epoch_single(
            model, valid_loader, device,
            is_train=False, target=target, modality=modality,
            optimizer=None, grad_clip=None, class_weights=class_weights
        )

        prev_lr = _get_lr(optimizer)
        if scheduler is not None:
            monitor_metric = va["loss"] if sched_mode == "min" else va["macro_f1"]
            scheduler.step(monitor_metric)
        cur_lr = _get_lr(optimizer)
        history["lr"].append(cur_lr)

        history["train_loss"].append(tr["loss"]); history["valid_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"]);   history["valid_acc"].append(va["acc"])
        history["train_macro_f1"].append(tr["macro_f1"]); history["valid_macro_f1"].append(va["macro_f1"])
        history["train_cls_acc"].append(tr["per_class_acc"]); history["valid_cls_acc"].append(va["per_class_acc"])

        def fmt_arr(a):
            return "[" + ", ".join("nan" if (isinstance(x, float) and (x != x)) else f"{float(x):.3f}" for x in a) + "]"
        print(
            f"[Epoch {ep:03d}] "
            f"TR loss {tr['loss']:.4f} | VA loss {va['loss']:.4f} | lr {cur_lr:.2e}\n"
            f"       TR acc {tr['acc']:.3f} | VA acc {va['acc']:.3f}\n"
            f"       TR F1  {tr['macro_f1']:.3f} | VA F1  {va['macro_f1']:.3f}\n"
            f"       TR per-class acc {fmt_arr(tr['per_class_acc'])}\n"
            f"       VA per-class acc {fmt_arr(va['per_class_acc'])}"
        )

        improved = (va["loss"] < best_val_score) if sched_mode == "min" else (va["macro_f1"] > best_val_score)
        if improved:
            best_val_score = va["loss"] if sched_mode == "min" else va["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {ep} (no improvement for {early_stopping_patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

def evaluate_on_loader_single(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    target: str = "severity",     # "severity" | "task"
    modality: str = "acc"         # "acc" | "traj"
) -> Dict[str, object]:
    """
    단일 모달(x_bag만 입력) 평가 함수.
    - target: "severity" 또는 "task"
    - modality: "acc" 또는 "traj"
    반환:
      {
        "acc": float,
        "macro_f1": float,
        "per_class_acc": List[float],
        "conf": torch.Tensor [C, C] (CPU)
      }
    """
    assert target in ("severity", "task"), "target must be 'severity' or 'task'"
    assert modality in ("acc", "traj"), "modality must be 'acc' or 'traj'"

    def _metrics_from_conf(conf: torch.Tensor):
        conf = conf.to(torch.float32)
        C = conf.size(0)
        support = conf.sum(dim=1)   # row sum (GT count)
        pred_cnt = conf.sum(dim=0)  # col sum (Pred count)
        tp = torch.diag(conf)

        per_class_acc = torch.full((C,), float('nan'), dtype=torch.float32)
        mask_sup = support > 0
        per_class_acc[mask_sup] = tp[mask_sup] / support[mask_sup]

        precision = torch.zeros(C, dtype=torch.float32)
        recall    = torch.zeros(C, dtype=torch.float32)
        precision[pred_cnt > 0] = tp[pred_cnt > 0] / pred_cnt[pred_cnt > 0]
        recall[mask_sup]        = tp[mask_sup] / support[mask_sup]

        denom = precision + recall
        f1 = torch.zeros(C, dtype=torch.float32)
        nz = denom > 0
        f1[nz] = 2 * precision[nz] * recall[nz] / denom[nz]

        macro_f1 = float(f1[mask_sup].mean().item()) if mask_sup.any() else 0.0
        return {"macro_f1": macro_f1, "per_class_acc": per_class_acc.tolist()}

    model.eval()

    epoch_conf = None  # [C, C] on CPU
    correct = 0.0
    total   = 0.0

    desc = f"Eval(single|{target}|{modality})"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        # --- 입력/전처리 ---
        if modality == "acc":
            x_bag = batch["acc_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True)  # [B,N,T,C]
        else:  # "traj"
            x_bag = batch["traj_bag"].permute(0, 1, 3, 2).to(device, non_blocking=True) # [B,N,T,C]

        mask = batch["mask"].to(device, non_blocking=True)  # [B,N] (bool)
        if target == "severity":
            y = batch["y"].to(device, non_blocking=True)
        else:
            y = batch["y_task"].to(device, non_blocking=True)

        # --- 모델 추론 ---
        out = model(x_bag, mask=mask)

        # 다양한 반환 형태를 견고하게 처리
        logits = out[0]

        # --- 메트릭 갱신 ---
        preds = logits.argmax(dim=-1)  # [B]
        y_cpu = y.detach().cpu()
        p_cpu = preds.detach().cpu()

        C = logits.size(-1)
        if epoch_conf is None:
            epoch_conf = torch.zeros(C, C, dtype=torch.long)

        for c in range(C):
            m = (y_cpu == c)
            if m.any():
                binc = torch.bincount(p_cpu[m], minlength=C)
                epoch_conf[c, :] += binc

        correct += (p_cpu == y_cpu).sum().item()
        total   += float(y_cpu.numel())

        # 진행 표시
        mm_now = _metrics_from_conf(epoch_conf)
        acc_now = correct / max(1.0, total)
        pbar.set_postfix({"acc": f"{acc_now:.3f}", "macro_f1": f"{mm_now['macro_f1']:.3f}"})

    if epoch_conf is None:
        return {"acc": 0.0, "macro_f1": 0.0, "per_class_acc": [], "conf": torch.zeros(0, 0)}

    acc_final = correct / max(1.0, total)
    mm = _metrics_from_conf(epoch_conf)

    return {
        "acc": float(acc_final),
        "macro_f1": float(mm["macro_f1"]),
        "per_class_acc": mm["per_class_acc"],
        "conf": epoch_conf,  # CPU tensor
    }
