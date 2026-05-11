import numpy as np
import pytest

from pofl.fl.pool_formation import (
    Algo3Step,
    TrainerProfile,
    algorithm3_pool_formation,
    partition_welfare,
    switch_gain,
)
from pofl.fl.welfare import (
    WelfareParams,
    federation_utility,
    global_label_distribution,
)


def _profile(
    tid: str,
    *,
    n: int = 50,
    delay: float = 1.0,
    label_bias: int = 0,
    num_classes: int = 4,
    seed: int = 0,
) -> TrainerProfile:
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(num_classes)
    logits[label_bias] += 2.0
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    labels = rng.choice(num_classes, size=n, p=p)
    return TrainerProfile(
        trainer_id=tid,
        sample_count=n,
        delay=delay,
        labels=labels,
        num_classes=num_classes,
    )


def _params() -> WelfareParams:
    return WelfareParams(train_time=200.0, psi_local_max=20)


def _make_population(n: int, seed: int) -> list[TrainerProfile]:
    return [
        _profile(
            f"m{i}",
            n=60,
            delay=1.0 + 0.05 * i,
            label_bias=i % 4,
            seed=seed * 100 + i,
        )
        for i in range(n)
    ]


def test_partition_covers_and_disjoint():
    profs = _make_population(8, seed=0)
    pools, _ = algorithm3_pool_formation(profs, _params(), rng=np.random.default_rng(1))
    seen = sorted([mid for po in pools for mid in po.member_ids])
    assert seen == sorted(p.trainer_id for p in profs)
    # Disjoint
    member_sets = [set(po.member_ids) for po in pools]
    for i in range(len(member_sets)):
        for j in range(i + 1, len(member_sets)):
            assert member_sets[i].isdisjoint(member_sets[j])


def test_terminates_within_max_iters():
    profs = _make_population(6, seed=2)
    pools, trace = algorithm3_pool_formation(
        profs, _params(), rng=np.random.default_rng(2), max_iters=20, trace=True
    )
    assert all(isinstance(s, Algo3Step) for s in trace)
    # Last accepted-switches count should be 0 (Nash-stable termination).
    assert trace[-1].n_switches == 0


def test_total_welfare_non_decreasing_across_iters():
    profs = _make_population(8, seed=3)
    _, trace = algorithm3_pool_formation(
        profs, _params(), rng=np.random.default_rng(3), trace=True
    )
    welfares = [s.total_welfare for s in trace]
    # Strictly: per Lemma 1, accepted switches yield positive gain, so
    # welfare must be non-decreasing iteration over iteration.
    for prev, nxt in zip(welfares, welfares[1:]):
        assert nxt + 1e-9 >= prev, f"welfare decreased: {prev} -> {nxt}"


def test_nash_stable_no_positive_unilateral_switch_at_termination():
    profs = _make_population(8, seed=4)
    params = _params()
    pools, _ = algorithm3_pool_formation(profs, params, rng=np.random.default_rng(4))
    # Reconstruct the partition as index lists.
    name_to_idx = {p.trainer_id: i for i, p in enumerate(profs)}
    idx_pools = [[name_to_idx[mid] for mid in po.member_ids] for po in pools]
    global_hist = global_label_distribution(profs)
    # No miner should have a strictly positive single-switch gain to any other
    # pool or to a new singleton.
    for src_pi, src in enumerate(idx_pools):
        for m in src:
            for dst_pi, dst in enumerate(idx_pools):
                if dst_pi == src_pi:
                    continue
                gain = switch_gain(m, src, dst, profs, global_hist, params)
                assert gain <= 1e-9, (
                    f"Nash violation: miner {m} can gain {gain} switching to pool {dst_pi}"
                )
            if len(src) > 1:
                gain = switch_gain(m, src, [], profs, global_hist, params)
                assert gain <= 1e-9, (
                    f"Nash violation: miner {m} can gain {gain} splitting off"
                )


def test_warm_start_respected_when_already_stable():
    profs = _make_population(4, seed=5)
    # Hand-craft a partition; check warm-start is at least the starting point.
    warm = [[0, 1], [2, 3]]
    pools, trace = algorithm3_pool_formation(
        profs, _params(), rng=np.random.default_rng(5), warm_start=warm, trace=True
    )
    assert trace[0].n_pools == 2
    # All miners present.
    seen = sorted([mid for po in pools for mid in po.member_ids])
    assert seen == sorted(p.trainer_id for p in profs)


def test_warm_start_validates_coverage():
    profs = _make_population(3, seed=6)
    with pytest.raises(ValueError):
        algorithm3_pool_formation(profs, _params(), warm_start=[[0, 1]])  # missing 2
    with pytest.raises(ValueError):
        algorithm3_pool_formation(profs, _params(), warm_start=[[0, 0, 1, 2]])  # dupe


def test_partition_welfare_matches_per_pool_sum():
    profs = _make_population(6, seed=7)
    params = _params()
    pools, _ = algorithm3_pool_formation(profs, params, rng=np.random.default_rng(7))
    # partition_welfare uses a global histogram across the full population.
    global_hist = global_label_distribution(
        [p for po in pools for p in po.profiles]
    )
    expected = sum(
        federation_utility(po.profiles, global_hist, params) for po in pools
    )
    assert partition_welfare(pools, params) == pytest.approx(expected)


def test_homogeneous_pool_beats_singletons():
    """When every miner is well-aligned with the global distribution, pooling
    is preferred over fragmentation (federation cost is dwarfed by the
    per-pool satisfaction gain from larger S_j)."""
    profs = [
        _profile(f"m{i}", n=200, delay=1.0, label_bias=0, num_classes=4, seed=i)
        for i in range(4)
    ]
    params = _params()
    pools, _ = algorithm3_pool_formation(
        profs, params, rng=np.random.default_rng(0)
    )
    # Algorithm should at least merge some miners (not stay all-singleton).
    assert any(len(po.member_ids) > 1 for po in pools)
