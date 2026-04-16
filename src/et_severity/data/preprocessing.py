"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *
from ..config import DIR_RE, SPLIT_FILE_RE, TASK2IDX

def split_and_delete_multidirect_fs(
    root: Path = Path("data"),
    *,
    c1_col: str = "c1",
    median_size: int = 5,
    suffix_map: Dict[int, str] = None,   # {100:"_1", 200:"_2", 300:"_3"}
    overwrite: bool = True
) -> dict:
    """
    data 하위에서 디렉토리명이 'MultiDirect'(대소문자 무시)인 모든 CSV를 찾아:
      1) c1 전처리 (round → abs → median_filter(size=median_size) → ×100)
      2) c1=={100,200,300} 행을 각각 _1/_2/_3 접미사로 새 CSV 저장
      3) 분할본이 하나라도 생성되면 원본 CSV 삭제

    이미 분할된 파일(…_0_1.csv 등, 끝에 '_<trial>_<1|2|3>.csv')은 재처리/삭제 대상에서 제외.

    Returns
    -------
    {'saved': [...], 'deleted': [...], 'kept': [...], 'errors': [...]}
    """
    if suffix_map is None:
        suffix_map = {100: "_1", 200: "_2", 300: "_3"}

    # MultiDirect 디렉토리 하위의 모든 CSV 수집 (대소문자 무시)
    csv_files: List[Path] = [
        p for p in root.rglob("*.csv")
        if any(part.lower() == "multidirect" for part in p.parts)
    ]

    saved, deleted, kept, errors = [], [], [], []
    seen = set()

    for p in sorted(csv_files):
        if p in seen:
            continue
        seen.add(p)

        # 이미 분할된 파일 패턴이면 스킵 (…_trialLevel_c1Level.csv)
        if SPLIT_FILE_RE.match(p.stem):
            kept.append(str(p))  # 그대로 유지
            continue

        # CSV 읽기
        try:
            df = pd.read_csv(p)
        except Exception as e:
            errors.append(f"[READ-ERROR] {p}: {e}")
            kept.append(str(p))
            continue

        if c1_col not in df.columns:
            errors.append(f"[MISSING-COLUMN] {p}: '{c1_col}' not found")
            kept.append(str(p))
            continue

        # c1 전처리: round → abs → median filter → ×100
        try:
            c1_raw = pd.to_numeric(df[c1_col], errors="coerce").to_numpy(dtype=float)
            c1_int = np.rint(c1_raw).astype(np.int64)
            c1_abs = np.abs(c1_int)
            c1_med = median_filter(c1_abs, size=median_size, mode="nearest").astype(np.int64)
            c1 = c1_med * 100  # 기대: {100,200,300}
        except Exception as e:
            errors.append(f"[C1-PROCESS-ERROR] {p}: {e}")
            kept.append(str(p))
            continue

        # 분할 저장
        wrote_any = False
        for lv, suf in suffix_map.items():
            m = (c1 == lv)
            if np.any(m):
                sub_df = df.loc[m].reset_index(drop=True)
                out_path = p.with_name(f"{p.stem}{suf}{p.suffix}")
                try:
                    if (not overwrite) and out_path.exists():
                        # 덮어쓰기 금지 시 존재하면 스킵
                        pass
                    else:
                        sub_df.to_csv(out_path, index=False)
                        saved.append(str(out_path))
                        wrote_any = True
                except Exception as e:
                    errors.append(f"[WRITE-ERROR] {out_path}: {e}")

        # 분할본이 하나라도 생겼다면 원본 삭제
        if wrote_any:
            try:
                p.unlink()
                deleted.append(str(p))
            except Exception as e:
                kept.append(str(p))
                errors.append(f"[DELETE-ERROR] {p}: {e}")
        else:
            kept.append(str(p))

    return {"saved": saved, "deleted": deleted, "kept": kept, "errors": errors}

def build_manifest(
    label_csv_path: Path,
    root_dir: Path,                  # 절대경로가 이미 들어있지만 인터페이스는 유지
    *,
    target_col = 'target',
    require_exists: bool = True,     # True면 실제 파일 존재하는 행만 남김
) -> pd.DataFrame:

    df = pd.read_csv(label_csv_path)

    required = {"file", "patient_id", "session", "task", target_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"다음 컬럼이 필요합니다: {missing}")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Build manifest (K=3)", unit="row"):
        # 1) 경로
        raw_file = str(r["file"]).strip()
        if not raw_file or raw_file.lower() == "nan":
            continue
        p = Path(raw_file)
        if not p.is_absolute():
            p = (root_dir / p).resolve()
        if require_exists and not p.exists():
            continue

        # 2) target (정수)
        try:
            target = int(r[target_col])
        except Exception:
            target = int(float(r[target_col]))

        # 3) patient_id: 'ET01' -> 1
        pid_match = re.search(r"(\d+)", str(r["patient_id"]))
        patient_id = int(pid_match.group(1)) if pid_match else -1

        # 4) session, date: 'ET01_1_01232017' → session=1, date='01232017'
        sess_str = str(r["session"])
        m = DIR_RE.search(sess_str) or DIR_RE.search(str(p))
        if m:
            session = int(m.group(2))
            date = m.group(3)
        else:
            session = -1
            date = "00000000"

        # 5) task & task_idx
        task = str(r["task"])
        task_idx = TASK2IDX.get(task.lower(), TASK2IDX["unknown"])

        rows.append(dict(
            path=p,
            patient_id=patient_id,
            session=session,
            date=date,
            task=task,
            task_idx=task_idx,
            target=target,
        ))

    man = pd.DataFrame(rows)
    if len(man):
        # [수정됨] target 컬럼이 0-based가 되도록 조정
        min_target = man["target"].min()
        if min_target > 0:
            man["target"] = man["target"] - min_target

        man = man.sort_values(["patient_id","session","task","path"]).reset_index(drop=True)
        man["target"] = man["target"].astype(int)
    return man

def split_train_valid_by_patient(
    manifest_df: pd.DataFrame,
    val_pid: Sequence[int],
    *,
    pid_col: str = "patient_id",
    reset_index: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    manifest 형태의 DataFrame을 받아, val_pid에 포함된 환자는 valid_df로,
    나머지는 train_df로 분리하여 반환합니다.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        'patient_id' 컬럼(기본값)을 포함한 manifest 데이터프레임.
    val_pid : Sequence[int]
        검증 셋에 넣을 환자 번호 리스트. 예) [2, 5, 9]
    pid_col : str, default 'patient_id'
        환자 ID가 들어있는 컬럼명.
    reset_index : bool, default True
        반환하는 데이터프레임의 인덱스를 0부터 재설정할지 여부.

    Returns
    -------
    train_df : pd.DataFrame
        val_pid에 속하지 않는 환자들의 샘플.
    valid_df : pd.DataFrame
        val_pid에 속하는 환자들의 샘플.
    """
    if pid_col not in manifest_df.columns:
        raise KeyError(f"'{pid_col}' 컬럼이 없습니다. 현재 컬럼: {list(manifest_df.columns)}")

    # 숫자형으로 안전 변환 (NaN은 검증 집합에 포함되지 않음)
    pid_series = pd.to_numeric(manifest_df[pid_col], errors="coerce")
    val_set = set(int(x) for x in val_pid)

    valid_mask = pid_series.isin(val_set)
    valid_df = manifest_df.loc[valid_mask].copy()
    train_df = manifest_df.loc[~valid_mask].copy()

    if reset_index:
        valid_df.reset_index(drop=True, inplace=True)
        train_df.reset_index(drop=True, inplace=True)

    return train_df, valid_df

def build_segment_manifest(
    manifest_df: pd.DataFrame,
    *,
    seg_len: int,
    hop: int,
    usecols: Sequence[str] = ("accel_x","accel_y","accel_z","coor_x","coor_y"),
    require_exists: bool = True,
    keep_tail: bool = False,   # 마지막 남는 길이를 버리지 않고 패딩해 포함하고 싶으면 True (패딩은 하지 않고 end=T로만 기록)
) -> pd.DataFrame:
    """
    Parameters
    ----------
    manifest_df : pd.DataFrame
        build_manifest() 결과. 필수 컬럼:
        ['path','patient_id','session','date','task','task_idx','target']
    seg_len : int
        세그먼트 길이(샘플 수)
    hop : int
        홉(겹침). 0 또는 음수면 seg_len으로 간주(오버랩 없음)
    usecols : Sequence[str]
        CSV에서 길이 확인을 위해 읽을 컬럼(존재하는 것만 사용)
    require_exists : bool
        True면 파일이 실제로 있는 행만 처리
    keep_tail : bool
        True면 (T - seg_len) % hop 잔여가 있을 때 마지막 구간을 하나 더 만듦
        (end는 T로 기록; 모델 입력에서 패딩은 사용자가 처리)

    Returns
    -------
    seg_manifest : pd.DataFrame
        컬럼: ['path','patient_id','session','date','task','task_idx','target','start','end','seg_idx']
    """
    required = {"path","patient_id","session","date","task","task_idx","target"}
    missing = required - set(manifest_df.columns)
    if missing:
        raise KeyError(f"다음 컬럼이 필요합니다: {missing}")

    rows = []
    H = seg_len if (hop is None or hop <= 0) else int(hop)

    for _, r in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Expand to segments", unit="file"):
        p = Path(str(r["path"]))
        if require_exists and not p.exists():
            continue

        # 존재하는 usecols만 선택해서 가볍게 읽기
        try:
            hdr = pd.read_csv(p, nrows=0)
            cols = [c for c in usecols if c in hdr.columns]
            if len(cols) == 0:
                # 컬럼명이 다르더라도 길이만 알면 되므로 전체 길이만 읽기
                T = len(pd.read_csv(p))
            else:
                tmp = pd.read_csv(p, usecols=cols)
                T = len(tmp)
        except Exception:
            continue

        if T <= 0:
            continue

        # 시작 인덱스들
        starts = list(range(0, max(0, T - seg_len + 1), H))
        if keep_tail and (T > seg_len):
            last_start = starts[-1] if len(starts) else 0
            last_end = last_start + seg_len
            if last_end < T:  # 잔여가 있으면 하나 더
                starts.append(T - seg_len)

        # 너무 짧으면 스킵
        if T < seg_len and not keep_tail:
            continue
        if T < seg_len and keep_tail:
            starts = [0]  # 하나만 만들고 end=T로 설정

        for j, s in enumerate(starts):
            start = int(s)
            end = int(min(s + seg_len, T))  # keep_tail=True일 때 T로 잘릴 수 있음
            rows.append(dict(
                path=str(p),
                patient_id=int(r["patient_id"]),
                session=int(r["session"]),
                date=str(r["date"]),
                task=str(r["task"]),
                task_idx=int(r["task_idx"]),
                target=int(r["target"]),
                start=start,
                end=end,
                seg_idx=j,
            ))

    seg = pd.DataFrame(rows)
    if len(seg):
        seg = seg.sort_values(["patient_id","session","task","path","start"]).reset_index(drop=True)
        # 타입 안전화
        for c in ["patient_id","session","task_idx","target","start","end","seg_idx"]:
            if c in seg.columns:
                seg[c] = seg[c].astype(int, errors="ignore")
    return seg

def expand_train_valid_to_segments(
    train_manifest: pd.DataFrame,
    valid_manifest: pd.DataFrame,
    *,
    seg_len: int,
    hop: int,
    usecols: Sequence[str] = ("accel_x","accel_y","accel_z","coor_x","coor_y"),
    require_exists: bool = True,
    keep_tail: bool = False,
):
    seg_train = build_segment_manifest(
        train_manifest, seg_len=seg_len, hop=hop, usecols=usecols,
        require_exists=require_exists, keep_tail=keep_tail,
    )
    seg_valid = build_segment_manifest(
        valid_manifest, seg_len=seg_len, hop=hop, usecols=usecols,
        require_exists=require_exists, keep_tail=keep_tail,
    )
    return seg_train, seg_valid
