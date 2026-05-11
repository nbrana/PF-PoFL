import math

import numpy as np
import pytest

from pofl.fl.pool_formation import TrainerProfile
from pofl.fl.welfare import (
    WelfareParams,
    cost,
    emd_to_global,
    expected_loss_bound,
    federation_utility,
    global_label_distribution,
    latency_admit,
    per_pool_max_rounds,
    pool_weighted_emd,
    relative_loss_E,
    satisfaction,
    total_welfare,
)


def _profile(
    tid: str, *, n: int, delay: float, labels: list[int], num_classes: int = 4
) -> TrainerProfile:
    return TrainerProfile(
        trainer_id=tid,
        sample_count=n,
        delay=delay,
        labels=np.array(labels, dtype=np.int64),
        num_classes=num_classes,
    )


def _params(**overrides) -> WelfareParams:
    base = dict(train_time=200.0, psi_local_max=20, beta=1.5)
    base.update(overrides)
    return WelfareParams(**base)


def test_relative_loss_E_zero_at_zero_emd():
    assert relative_loss_E(0.0) == 1.0
    assert relative_loss_E(0.5) == 1.5
    assert relative_loss_E(-0.1) == 1.0  # clamped at 0


def test_emd_to_global_l1():
    p = np.array([0.5, 0.5, 0.0, 0.0])
    g = np.array([0.25, 0.25, 0.25, 0.25])
    assert emd_to_global(p, g) == pytest.approx(1.0)


def test_pool_weighted_emd_sample_weighted():
    p1 = _profile("a", n=100, delay=1.0, labels=[0] * 100)  # all label 0
    p2 = _profile("b", n=10, delay=1.0, labels=[1] * 10)  # all label 1
    g = np.array([0.5, 0.5, 0.0, 0.0])
    bar = pool_weighted_emd([p1, p2], g)
    # weighted by sample count: 100/110 * EMD(p1,g) + 10/110 * EMD(p2,g)
    assert bar == pytest.approx((100 / 110) * 1.0 + (10 / 110) * 1.0)


def test_satisfaction_strictly_decreasing_in_pi():
    params = _params()
    s_low = satisfaction(0.01, params)
    s_high = satisfaction(0.5, params)
    assert s_low > s_high > 0


def test_satisfaction_zero_for_inf_loss():
    assert satisfaction(math.inf, _params()) == 0.0


def test_cost_linear_and_zero_for_singleton():
    params = _params()
    assert cost(1, params) == 0.0
    assert cost(2, params) == params.lambda_c * 2
    assert cost(5, params) == params.lambda_c * 5


def test_global_label_distribution_weighted():
    p1 = _profile("a", n=100, delay=1.0, labels=[0] * 100, num_classes=2)
    p2 = _profile("b", n=300, delay=1.0, labels=[1] * 300, num_classes=2)
    g = global_label_distribution([p1, p2])
    assert g.shape == (2,)
    assert g[0] == pytest.approx(0.25)
    assert g[1] == pytest.approx(0.75)


def test_latency_admit_threshold():
    p_fast = _profile("f", n=10, delay=1.0, labels=[0] * 10)
    p_slow = _profile("s", n=10, delay=10.0, labels=[0] * 10)
    assert latency_admit(p_fast, d_max=2.0) is True
    assert latency_admit(p_slow, d_max=2.0) is False


def test_per_pool_max_rounds_nonzero():
    p = _profile("a", n=10, delay=1.0, labels=[0] * 10)
    psi = per_pool_max_rounds([p], train_time=200.0, beta=1.5)
    assert psi >= 1


def test_expected_loss_bound_singleton_uses_psi_local():
    params = _params(psi_local_max=20)
    p = _profile("a", n=100, delay=1.0, labels=[0] * 100)
    g = global_label_distribution([p])  # so EMD(p,g) == 0
    pi_singleton = expected_loss_bound([p], g, params)
    # E(0)=1; loss bound = 1/sqrt(100*20) + 1/20
    assert pi_singleton == pytest.approx(1.0 / math.sqrt(100 * 20) + 1.0 / 20)


def test_expected_loss_bound_pool_uses_psi_global():
    params = _params(train_time=200.0, beta=1.5)
    p1 = _profile("a", n=100, delay=1.0, labels=[0] * 100)
    p2 = _profile("b", n=100, delay=1.0, labels=[0] * 100)
    g = global_label_distribution([p1, p2])  # EMD == 0 for both
    pi = expected_loss_bound([p1, p2], g, params)
    psi_global = per_pool_max_rounds([p1, p2], params.train_time, params.beta)
    assert pi == pytest.approx(1.0 / math.sqrt(200 * psi_global) + 1.0 / psi_global)


def test_expected_loss_bound_inf_when_all_excluded_by_latency():
    params = _params(beta=0.1)  # tight cap
    p = _profile("slow", n=100, delay=1000.0, labels=[0] * 100)
    p2 = _profile("slow2", n=100, delay=1000.0, labels=[0] * 100)
    g = global_label_distribution([p, p2])
    assert math.isinf(expected_loss_bound([p, p2], g, params))


def test_utility_increases_with_aligned_high_sample_miner():
    """Adding a low-EMD high-sample miner to a pool should raise federation utility."""
    params = _params()
    base = [_profile(f"m{i}", n=100, delay=1.0, labels=[0] * 50 + [1] * 50) for i in range(2)]
    aligned = _profile("good", n=200, delay=1.0, labels=[0] * 100 + [1] * 100)
    g = global_label_distribution(base + [aligned])
    u_before = federation_utility(base, g, params)
    u_after = federation_utility(base + [aligned], g, params)
    assert u_after > u_before


def test_utility_falls_when_high_latency_miner_added():
    """High-latency miner is excluded by alpha_m=0; cost grows with pool size -> utility falls."""
    params = _params(beta=1.0)
    base = [
        _profile(f"m{i}", n=100, delay=1.0, labels=[0] * 50 + [1] * 50) for i in range(3)
    ]
    g = global_label_distribution(base)
    bad = _profile("slow", n=100, delay=999.0, labels=[0] * 50 + [1] * 50)
    u_before = federation_utility(base, g, params)
    u_after = federation_utility(base + [bad], g, params)
    assert u_after < u_before


def test_total_welfare_sums_pools():
    params = _params()
    p1 = _profile("a", n=80, delay=1.0, labels=[0] * 40 + [1] * 40)
    p2 = _profile("b", n=80, delay=1.0, labels=[0] * 40 + [1] * 40)
    g = global_label_distribution([p1, p2])
    w = total_welfare([[p1], [p2]], g, params)
    assert w == pytest.approx(
        federation_utility([p1], g, params) + federation_utility([p2], g, params)
    )
