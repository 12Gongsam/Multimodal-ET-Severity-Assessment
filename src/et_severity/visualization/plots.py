"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *

def add_sig_bracket(x1, x2, y, h=2.0, text='*'):
    plt.plot([x1, x1, x2, x2],
             [y,  y+h, y+h, y],
             color='black', linewidth=1.0)
    plt.text((x1 + x2) / 2.0,
             y + h + 0.5,
             text,
             ha='center', va='bottom', fontsize=11)

def plot_gmm_single(
    df,
    column,
    *,
    k=3,
    bins=30,
    n_init=100,
    random_state=42,
    grid_points=1000,
    covariance_type='full',
    ax=None,
    reset_cycle=True
):
    """
    단일 수치 컬럼(1D)에 대해 지정한 k 값으로 GMM 적합/시각화.
    (컴포넌트 레이블이 왼쪽부터 1, 2, 3...이 되도록 정렬)
    """
    # --- 데이터 준비 ---
    x = df[column].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max(3, k):
        raise ValueError(f"유효 표본 {x.size}개: 최소 max(3, k)={max(3,k)} 이상 필요합니다.")
    X = x.reshape(-1, 1)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_min == x_max:
        eps = max(1e-6, 0.01 * max(1.0, abs(x_min)))
        x_min, x_max = x_min - eps, x_max + eps
    x_range = np.linspace(x_min, x_max, grid_points).reshape(-1, 1)

    def _comp_std(gmm, i):
        ct = gmm.covariance_type
        if ct == 'full':
            var = gmm.covariances_[i][0, 0]
        elif ct == 'diag':
            var = gmm.covariances_[i][0]
        elif ct == 'spherical':
            var = gmm.covariances_[i]
        elif ct == 'tied':
            var = gmm.covariances_[0, 0]
        else:
            raise ValueError(f"지원되지 않는 covariance_type: {ct}")
        return np.sqrt(var)

    # --- GMM 적합 ---
    gmm = GaussianMixture(
        n_components=k,
        n_init=n_init,
        random_state=random_state,
        covariance_type=covariance_type
    ).fit(X)

    # --- 플로팅 ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        created_fig = True

    # (1) 히스토그램
    ax.hist(x, bins=bins, density=True, color='skyblue', alpha=0.6, label='Data Histogram')

    # (2) 전체 혼합분포 (검정 점선)
    log_prob = gmm.score_samples(x_range)
    pdf = np.exp(log_prob)
    ax.plot(x_range, pdf, color='black', linestyle='--', label='Overall GMM Fit')

    # (3) 컴포넌트: 색상 순환 초기화로 일관성 유지
    if reset_cycle:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
        if default_colors:
            ax.set_prop_cycle(cycler(color=default_colors))

    # --- 정렬된 컴포넌트 플롯 ---
    sorted_indices = np.argsort(gmm.means_.ravel())
    for label_idx, original_idx in enumerate(sorted_indices):
        mu = gmm.means_[original_idx, 0]
        sd = _comp_std(gmm, original_idx)
        w = gmm.weights_[original_idx]
        comp_pdf = w * norm.pdf(x_range.ravel(), loc=mu, scale=sd)
        ax.plot(x_range, comp_pdf, label=f'Component {label_idx + 1}')

    # 라벨/범례
    ax.set_xlabel('Log Power', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend()

    # ── 요청한 스타일 적용: 상/우 스파인 숨김 + 틱 지정 ──
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([-5, -4, -3, -2])
    ax.tick_params(axis='both', labelsize=14)
    
    plt.show()

    return gmm, ax


add_significance_bracket = add_sig_bracket
plot_gaussian_mixture_fit = plot_gmm_single
