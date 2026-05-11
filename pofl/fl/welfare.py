"""Federation utility, satisfaction, and cost functions (paper Sec. V-A, eqs. 21-28).

All formulas trace to Wang et al. PF-PoFL. Function names cite the equation
they implement so reviewers can follow the proofs back to the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from pofl.fl.emd import label_histogram

if TYPE_CHECKING:
    from pofl.fl.pool_formation import TrainerProfile

GAMMA_S = 23.0
GAMMA_D = 30.0
LAMBDA_C = 0.01
BETA_DELAY = 1.5


@dataclass(frozen=True)
class WelfareParams:
    train_time: float
    psi_local_max: int
    beta: float = BETA_DELAY
    gamma_s: float = GAMMA_S
    gamma_d: float = GAMMA_D
    lambda_c: float = LAMBDA_C


def global_label_distribution(profiles: Sequence[TrainerProfile]) -> np.ndarray:
    """Sample-weighted label distribution over the whole task population."""
    if not profiles:
        raise ValueError("global_label_distribution: empty profiles")
    num_classes = profiles[0].num_classes
    acc = np.zeros(num_classes, dtype=np.float64)
    total = 0
    for p in profiles:
        h = label_histogram(p.labels, p.num_classes)
        acc += h * p.sample_count
        total += p.sample_count
    if total == 0:
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    return acc / total


def emd_to_global(p_hist: np.ndarray, global_hist: np.ndarray) -> float:
    """Paper eq. 21: L1 distance between miner label dist and global dist."""
    if p_hist.shape != global_hist.shape:
        raise ValueError("emd_to_global: shape mismatch")
    return float(np.abs(p_hist - global_hist).sum())


def pool_weighted_emd(
    profiles: Sequence[TrainerProfile], global_hist: np.ndarray
) -> float:
    """Paper eq. 22: sample-weighted mean of per-miner EMD-to-global."""
    if not profiles:
        return 0.0
    total = sum(p.sample_count for p in profiles)
    if total == 0:
        return 0.0
    acc = 0.0
    for p in profiles:
        h = label_histogram(p.labels, p.num_classes)
        acc += (p.sample_count / total) * emd_to_global(h, global_hist)
    return float(acc)


def relative_loss_E(emd: float) -> float:
    """Paper Fig. 9 surrogate: linear in EMD with E(0) = 1."""
    return 1.0 + max(0.0, float(emd))


def latency_admit(profile: TrainerProfile, d_max: float) -> bool:
    """Paper eq. 24: alpha_m = 1 iff D_nl,m <= D_j^max."""
    return float(profile.delay) <= float(d_max)


def per_pool_max_rounds(
    profiles: Sequence[TrainerProfile], train_time: float, beta: float = BETA_DELAY
) -> int:
    """Paper Psi_global = floor(T_train / D_j^max), with D_j^max = beta * mean delay."""
    if not profiles or train_time <= 0:
        return 1
    mean_delay = float(np.mean([p.delay for p in profiles]))
    d_max = beta * max(mean_delay, 1e-9)
    return max(1, int(math.floor(train_time / d_max)))


def expected_loss_bound(
    profiles: Sequence[TrainerProfile],
    global_hist: np.ndarray,
    params: WelfareParams,
) -> float:
    """Paper eq. 25: Pi_j accuracy-loss bound. Splits |J|>1 vs |J|=1."""
    if not profiles:
        return math.inf
    if len(profiles) == 1:
        m = profiles[0]
        h = label_histogram(m.labels, m.num_classes)
        emd_m = emd_to_global(h, global_hist)
        s_m = max(int(m.sample_count), 1)
        psi_local = max(int(params.psi_local_max), 1)
        return relative_loss_E(emd_m) * (
            1.0 / math.sqrt(s_m * psi_local) + 1.0 / psi_local
        )

    mean_delay = float(np.mean([p.delay for p in profiles]))
    d_max = params.beta * max(mean_delay, 1e-9)
    admitted = [p for p in profiles if latency_admit(p, d_max)]
    if not admitted:
        return math.inf

    s_pool = sum(p.sample_count for p in admitted)
    if s_pool <= 0:
        return math.inf
    psi_global = max(per_pool_max_rounds(profiles, params.train_time, params.beta), 1)
    emd_bar = pool_weighted_emd(admitted, global_hist)
    e_emd = relative_loss_E(emd_bar)
    return e_emd * (1.0 / math.sqrt(s_pool * psi_global) + 1.0 / psi_global)


def satisfaction(pi_j: float, params: WelfareParams) -> float:
    """Paper eq. 26: S(Pi_j) = gamma_s * exp(-gamma_d * Pi_j)."""
    if math.isinf(pi_j):
        return 0.0
    return float(params.gamma_s * math.exp(-params.gamma_d * pi_j))


def cost(pool_size: int, params: WelfareParams) -> float:
    """Paper eq. 27: lambda_c * |J| if |J|>1 else 0."""
    if pool_size <= 1:
        return 0.0
    return float(params.lambda_c * pool_size)


def federation_utility(
    profiles: Sequence[TrainerProfile],
    global_hist: np.ndarray,
    params: WelfareParams,
) -> float:
    """Paper eq. 28: U(J_j) = S(Pi_j) - C(J_j)."""
    if not profiles:
        return 0.0
    pi_j = expected_loss_bound(profiles, global_hist, params)
    return satisfaction(pi_j, params) - cost(len(profiles), params)


def total_welfare(
    partition: Sequence[Sequence[TrainerProfile]],
    global_hist: np.ndarray,
    params: WelfareParams,
) -> float:
    """Paper eq. 29: W(J) = sum over pools of federation utility."""
    return float(
        sum(federation_utility(list(pool), global_hist, params) for pool in partition)
    )
