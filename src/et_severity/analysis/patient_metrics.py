"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *

def compute_patient_metrics(
    df: pd.DataFrame,
    *,
    id_col: str = "patient_id",
    true_col: str = "true_severity",
    pred_col: str = "pred_severity",
    drop_na: bool = True,
    auto_detect: bool = True,
) -> pd.DataFrame:
    """
    df에서 patient_id 별로 Accuracy / Weighted Precision / Weighted Recall / Weighted F1 계산.
    OVERALL 행은 포함하지 않습니다.

    - 기본은 severity 컬럼을 사용(true_severity/pred_severity).
    - auto_detect=True 인 경우, 지정된 컬럼이 없으면 task 컬럼(true_task/pred_task)로 자동 대체.
    - 라벨이 숫자형 or 숫자 문자열이면 int로, 그 외는 문자열로 통일.
    """

    data = df.copy()

    # 1) 컬럼 자동 감지(필요 시)
    if auto_detect:
        need = {id_col, true_col, pred_col} - set(data.columns)
        if need:
            # severity 기본값이 없으면 task 쌍을 시도
            if {"true_task", "pred_task"}.issubset(data.columns):
                true_col, pred_col = "true_task", "pred_task"
            else:
                # 그래도 없으면 에러
                missing = {id_col, true_col, pred_col} - set(data.columns)
                if missing:
                    raise KeyError(f"다음 컬럼이 필요합니다: {sorted(missing)}")
    else:
        # auto_detect 사용하지 않을 때는 명시 컬럼 검사
        missing = {id_col, true_col, pred_col} - set(data.columns)
        if missing:
            raise KeyError(f"다음 컬럼이 필요합니다: {sorted(missing)}")

    # 2) 결측치 처리
    if drop_na:
        data = data.dropna(subset=[id_col, true_col, pred_col])

    # 3) 라벨 타입 통일: 숫자/숫자문자열 → int, 그 외 → str
    def _coerce_labels(s: pd.Series) -> pd.Series:
        if is_numeric_dtype(s):
            return s.astype(int)
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().all():      # 전부 숫자 문자열인 경우
            return s_num.astype(int)
        return s.astype(str)         # 혼합/문자 라벨은 문자열로

    data[true_col] = _coerce_labels(data[true_col])
    data[pred_col] = _coerce_labels(data[pred_col])

    # 4) 환자별 메트릭 계산
    rows = []
    for pid, g in data.groupby(id_col, sort=True):
        y_true = g[true_col].values
        y_pred = g[pred_col].values
        n = len(g)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        rows.append([pid, n, acc, prec, rec, f1])

    result = pd.DataFrame(
        rows, columns=[id_col, "n", "accuracy", "precision_w", "recall_w", "f1_w"]
    )
    return result

def analyze_ordinal_agreement(df, col_true, col_pred):
    """
    순서형 변수 간의 통계적 비교 및 시각화
    col_true: 정답/기준 컬럼 (예: tetras_score)
    col_pred: 예측/비교 컬럼 (예: gmm_pred)
    """
    # 데이터 준비 (결측치 제거)
    clean_df = df[[col_true, col_pred]].dropna()
    y_true = clean_df[col_true]
    y_pred = clean_df[col_pred]

    print(f"--- 분석 대상: {col_true} vs {col_pred} (n={len(clean_df)}) ---")

    # 1. 순위 상관계수 (Rank Correlation)
    # Spearman: 일반적인 순위 상관관계
    rho, p_spearman = stats.spearmanr(y_true, y_pred)
    # Kendall's Tau: 동점(ties)이 많을 때 더 견고함 (점수가 0~4로 제한적이므로 적합)
    tau, p_kendall = stats.kendalltau(y_true, y_pred)

    print(f"\n1. 순위 상관분석")
    print(f"   - Spearman Correlation: {rho:.4f} (p={p_spearman:.4f})")
    print(f"   - Kendall's Tau       : {tau:.4f} (p={p_kendall:.4f})")

    # 2. 일치도 분석 (Weighted Kappa)
    # weights='quadratic': 틀린 정도(거리)에 가중치 부여 (1점 차이 < 3점 차이)
    # 임상 척도 비교에서 표준으로 많이 사용됨 (QWK)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

    print(f"\n2. 일치도 분석 (Quadratic Weighted Kappa)")
    print(f"   - QWK Score: {qwk:.4f}")
    print("     (0.8+ : 아주 좋음, 0.6~0.8 : 좋음, 0.4~0.6 : 보통)")

    # 3. Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))

    # 교차표 생성
    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel(f'GMM Prediction', fontsize=16)
    plt.ylabel(f'TETRAS score', fontsize=16)
    plt.show()

def mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med))
