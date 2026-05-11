"""Distributed pool formation via the FFG-TU game (paper Sec. V-B, Algorithm 3).

Replaces the older centralized greedy / iterative best-response heuristics.
The algorithm here implements the paper's two-phase per-iteration loop:

  Phase 2 (miner side):
    Each miner m computes its switchable candidate set C_m^pool, picks the
    destination pool with the largest switch gain Omega_m, and sends a
    transfer request.

  Phase 3 (pool side):
    Each destination pool admits only the requester with the largest switch
    gain (paper Definition 8) and rejects the rest.

Iterates until a full sweep yields zero accepted switches (Nash-stable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pofl.fl.welfare import (
    WelfareParams,
    federation_utility,
    global_label_distribution,
    total_welfare,
)


@dataclass
class TrainerProfile:
    trainer_id: str
    sample_count: int
    delay: float
    labels: np.ndarray
    num_classes: int


@dataclass
class MiningPool:
    pool_id: str
    member_ids: list[str]
    profiles: list[TrainerProfile]


@dataclass(frozen=True)
class Algo3Step:
    iteration: int
    n_pools: int
    n_switches: int
    total_welfare: float


def switch_gain(
    miner_idx: int,
    src_indices: Sequence[int],
    dst_indices: Sequence[int],
    profiles: Sequence[TrainerProfile],
    global_hist: np.ndarray,
    params: WelfareParams,
) -> float:
    """Paper Definition 4 (eq. 31): Omega_m(Phi_l,k).

    Omega = [U(J_k ∪ {m}) − U(J_k)] − [U(J_l) − U(J_l\\{m})].

    `src_indices` may include `miner_idx` (current pool); `dst_indices` is the
    candidate destination pool (may be empty -> singleton formation).
    Returns 0 when src == dst (paper Remark 3 special case).
    """
    if list(src_indices) == list(dst_indices):
        return 0.0

    src_with = [profiles[i] for i in src_indices]
    src_without = [profiles[i] for i in src_indices if i != miner_idx]
    dst_without = [profiles[i] for i in dst_indices]
    dst_with = dst_without + [profiles[miner_idx]]

    u_src_with = federation_utility(src_with, global_hist, params) if src_with else 0.0
    u_src_without = (
        federation_utility(src_without, global_hist, params) if src_without else 0.0
    )
    u_dst_without = (
        federation_utility(dst_without, global_hist, params) if dst_without else 0.0
    )
    u_dst_with = federation_utility(dst_with, global_hist, params) if dst_with else 0.0

    return (u_dst_with - u_dst_without) - (u_src_with - u_src_without)


def _initial_partition(
    n_miners: int, warm_start: Optional[Sequence[Sequence[int]]]
) -> list[list[int]]:
    if warm_start is None:
        return [[i] for i in range(n_miners)]
    seen: set[int] = set()
    pools: list[list[int]] = []
    for pool in warm_start:
        block = [int(i) for i in pool]
        for i in block:
            if i in seen or not (0 <= i < n_miners):
                raise ValueError(f"warm_start invalid index or duplicate: {i}")
            seen.add(i)
        if block:
            pools.append(block)
    if seen != set(range(n_miners)):
        raise ValueError("warm_start does not cover all miners")
    return pools


def _pool_id_for(indices: Sequence[int], profiles: Sequence[TrainerProfile]) -> str:
    if not indices:
        return "pool-empty"
    sorted_ids = sorted(profiles[i].trainer_id for i in indices)
    return "pool-" + "+".join(sorted_ids)


def algorithm3_pool_formation(
    profiles: Sequence[TrainerProfile],
    params: WelfareParams,
    *,
    max_iters: int = 50,
    warm_start: Optional[Sequence[Sequence[int]]] = None,
    rng: Optional[np.random.Generator] = None,
    trace: bool = False,
) -> tuple[list[MiningPool], list[Algo3Step]]:
    """Paper Algorithm 3 (Sec. V-B): distributed FFG-TU pool formation.

    Returns the final partition as a list of MiningPool plus an optional
    per-iteration trace (Fig. 15 data: number of pools and number of
    accepted switches).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(profiles)
    if n == 0:
        return [], []

    global_hist = global_label_distribution(profiles)
    pools = _initial_partition(n, warm_start)
    history: list[Algo3Step] = []

    if trace:
        history.append(
            Algo3Step(
                iteration=0,
                n_pools=len(pools),
                n_switches=0,
                total_welfare=total_welfare(
                    [[profiles[i] for i in pool] for pool in pools],
                    global_hist,
                    params,
                ),
            )
        )

    for h in range(1, max_iters + 1):
        # Phase 2: collect transfer requests on miner side.
        # Map: dst_pool_idx (-1 for "new singleton") -> list[(gain, miner_idx)].
        requests: dict[int, list[tuple[float, int]]] = {}
        miner_order = list(range(n))
        rng.shuffle(miner_order)

        miner_to_pool = [-1] * n
        for pi, pool in enumerate(pools):
            for m in pool:
                miner_to_pool[m] = pi

        for m in miner_order:
            src_pi = miner_to_pool[m]
            best_gain = 0.0
            best_dst: int | None = None  # None == stay; -1 == new singleton
            # Candidate: each existing pool != current.
            for pj, dst in enumerate(pools):
                if pj == src_pi:
                    continue
                gain = switch_gain(
                    m, pools[src_pi], dst, profiles, global_hist, params
                )
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_dst = pj
            # Candidate: new singleton (only meaningful if current pool size > 1).
            if len(pools[src_pi]) > 1:
                gain = switch_gain(
                    m, pools[src_pi], [], profiles, global_hist, params
                )
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_dst = -1
            if best_dst is not None:
                requests.setdefault(best_dst, []).append((best_gain, m))

        if not requests:
            if trace:
                history.append(
                    Algo3Step(
                        iteration=h,
                        n_pools=len(pools),
                        n_switches=0,
                        total_welfare=total_welfare(
                            [[profiles[i] for i in pool] for pool in pools],
                            global_hist,
                            params,
                        ),
                    )
                )
            break

        # Phase 3: each destination pool accepts the requester with the largest
        # gain (paper Definition 8). To preserve Lemma 1's monotonicity (welfare
        # non-decreasing across iterations), no source or destination pool may
        # participate in more than one accepted move per iteration -- otherwise
        # two miners could "swap" pools, each with positive snapshot gain but
        # zero net welfare change. Resolve conflicts by greedy gain order.
        per_dst_best: list[tuple[float, int, int]] = []  # (gain, miner, dst)
        for dst, candidates in requests.items():
            candidates.sort(key=lambda t: (-t[0], t[1]))
            best_gain, best_m = candidates[0]
            per_dst_best.append((best_gain, best_m, dst))
        per_dst_best.sort(key=lambda t: (-t[0], t[1], t[2]))
        touched: set[int] = set()
        accepted: list[tuple[int, int]] = []  # (miner_idx, dst_pool_idx_or_-1)
        for gain, m, dst in per_dst_best:
            src_pi = miner_to_pool[m]
            if src_pi in touched:
                continue
            if dst != -1 and dst in touched:
                continue
            touched.add(src_pi)
            if dst != -1:
                touched.add(dst)
            accepted.append((m, dst))

        # Apply moves atomically (paper: split src then merge dst).
        # Sort accepted by miner_idx to make removal stable.
        new_pools = [list(pool) for pool in pools]
        # Each miner appears at most once in accepted (it requested at most one
        # destination), so we can apply sequentially.
        for m, dst in accepted:
            for pool in new_pools:
                if m in pool:
                    pool.remove(m)
                    break
            if dst == -1:
                new_pools.append([m])
            else:
                new_pools[dst].append(m)
        # Drop any pools emptied by the splits.
        new_pools = [p for p in new_pools if p]
        pools = new_pools

        if trace:
            history.append(
                Algo3Step(
                    iteration=h,
                    n_pools=len(pools),
                    n_switches=len(accepted),
                    total_welfare=total_welfare(
                        [[profiles[i] for i in pool] for pool in pools],
                        global_hist,
                        params,
                    ),
                )
            )

    out: list[MiningPool] = []
    for pool in pools:
        prof = [profiles[i] for i in pool]
        out.append(
            MiningPool(
                pool_id=_pool_id_for(pool, profiles),
                member_ids=[profiles[i].trainer_id for i in pool],
                profiles=prof,
            )
        )
    return out, history


def partition_welfare(
    pools: Sequence[MiningPool], params: WelfareParams
) -> float:
    """Paper eq. 29: total social welfare W(J)."""
    if not pools:
        return 0.0
    all_profiles = [p for pool in pools for p in pool.profiles]
    global_hist = global_label_distribution(all_profiles)
    return total_welfare([list(pool.profiles) for pool in pools], global_hist, params)
