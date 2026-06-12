"""Sensor augmentations used by MIL datasets."""

import math
from typing import Optional

import numpy as np
import torch

def _rand_unit_vec3(rng):
    v = rng.standard_normal(3)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _rot_R3_axis_angle(axis, theta):
    x,y,z = axis
    c, s = math.cos(theta), math.sin(theta)
    C = 1 - c
    # Rodrigues' rotation formula
    return np.array([
        [c+x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c+y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c+z*z*C]
    ], dtype=np.float32)

def _rot_R2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,-s],[s,c]], dtype=np.float32)

class BagRotator:
    """
    bag 전체에 동일 회전(라벨 보존 관점에서 센서/좌표계 변화만 반영)
    - acc_bag: [N,3,T], traj_bag: [N,2,T] (torch.float32)
    - prob: 회전 적용 확률 (기본 0.5)
    """
    def __init__(self, seed=123, prob: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.prob = float(prob)

    def __call__(
        self,
        acc_bag: Optional[torch.Tensor] = None,
        traj_bag: Optional[torch.Tensor] = None,
    ):
        if acc_bag is None and traj_bag is None:
            raise ValueError("At least one modality must be provided.")

        if self.rng.random() < self.prob:
            if acc_bag is not None:
                axis = _rand_unit_vec3(self.rng)
                theta3 = self.rng.uniform(-math.pi, math.pi)
                R3 = torch.from_numpy(_rot_R3_axis_angle(axis, theta3)).to(acc_bag)
                acc_bag = torch.einsum("ij,njt->nit", R3, acc_bag)

            if traj_bag is not None:
                theta2 = self.rng.uniform(-math.pi, math.pi)
                R2 = torch.from_numpy(_rot_R2(theta2)).to(traj_bag)
                traj_bag = torch.einsum("ij,njt->nit", R2, traj_bag)

        return acc_bag, traj_bag

def _permute_chunks(x: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    x: (C,T). N개 동일 길이 청크로 쪼개 무작위 순열. 마지막 청크는 잔여 포함.
    """
    C, T = x.shape
    if N <= 1: return x
    L = T // N
    idxs = [slice(i*L, (i+1)*L) for i in range(N-1)] + [slice((N-1)*L, T)]
    order = np.arange(N); rng.shuffle(order)
    return np.concatenate([x[:, idxs[i]] for i in order], axis=1)

def _timewarp_indices(T: int, rng: np.random.Generator,
                      a_range=(0.15, 0.35), f_range=(0.5, 1.5)):
    """
    속도곡선 v(t)=1 + a*sin(2π f t/T + φ), 누적으로 단조증가 τ(t) 생성 후 0..T-1로 정규화
    반환: τ (length T, strictly increasing)
    """
    t = np.arange(T, dtype=np.float32)
    a = rng.uniform(*a_range)          # amplitude of speed
    f = rng.uniform(*f_range)          # frequency (cycles per window)
    phi = rng.uniform(0, 2*math.pi)    # phase
    v = 1.0 + a * np.sin(2*math.pi*f*t/T + phi)
    v = np.maximum(v, 1e-3)            # positive speed
    tau = np.cumsum(v)
    tau = (tau - tau[0]) / (tau[-1] - tau[0]) * (T - 1)
    return tau.astype(np.float32)

def _interp1(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """ x: (C,T), tau: (T,) old->new 매핑에서 x_new[t]=x_old[tau[t]] """
    C, T = x.shape
    t = np.arange(T, dtype=np.float32)
    y = np.empty_like(x)
    for c in range(C):
        y[c] = np.interp(t, tau, x[c])
    return y

class InstancePermTimeW:
    """
    인스턴스 단위 Perm + TimeW (가속도/궤적에 동일 연산으로 동기 유지)
    - Perm: N in [n_min..n_max], 확률 prob_perm로 적용
    - TimeW: 확률 prob_timew로 적용
    """
    def __init__(self, seed=42, n_min=1, n_max=5,
                 tw_a=(0.15,0.35), tw_f=(0.5,1.5),
                 apply_perm=True, apply_timew=True,
                 prob_perm: float = 0.5, prob_timew: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.n_min = n_min
        self.n_max = n_max
        self.tw_a = tw_a
        self.tw_f = tw_f
        self.apply_perm = apply_perm
        self.apply_timew = apply_timew
        self.prob_perm = float(prob_perm)
        self.prob_timew = float(prob_timew)

    def __call__(
        self,
        acc_3xt: Optional[np.ndarray] = None,
        traj_2xt: Optional[np.ndarray] = None,
    ):
        if acc_3xt is None and traj_2xt is None:
            raise ValueError("At least one modality must be provided.")

        a = None if acc_3xt is None else acc_3xt.copy()
        t = None if traj_2xt is None else traj_2xt.copy()
        reference = a if a is not None else t
        T = reference.shape[1]

        # 1) Permutation (동일 순서 적용)
        do_perm = self.apply_perm and (self.rng.random() < self.prob_perm)
        if do_perm:
            N = int(self.rng.integers(self.n_min, self.n_max + 1))
            if N > 1:
                L = T // N
                idxs = [slice(i*L, (i+1)*L) for i in range(N-1)] + [slice((N-1)*L, T)]
                order = np.arange(N)
                self.rng.shuffle(order)  # 동일 order를 a,t에 같이 사용
                def _perm(x):
                    if x is None:
                        return None
                    return np.concatenate([x[:, idxs[i]] for i in order], axis=1)
                a = _perm(a)
                t = _perm(t)

        # 2) TimeWarp (동일 tau 적용)
        do_timew = self.apply_timew and (self.rng.random() < self.prob_timew)
        if do_timew:
            tau = _timewarp_indices(T, self.rng, a_range=self.tw_a, f_range=self.tw_f)
            if a is not None:
                a = _interp1(a, tau)
            if t is not None:
                t = _interp1(t, tau)

        return a, t
