#!/usr/bin/env python3
"""Light evaluation sweep + trivial spoof/Sybil probes for PF-PoFL PoC metrics."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pofl.economics import redeem_deposit, slash_fraction, split_hosting_fee  # noqa: E402
from pofl.fl.pool_formation import (  # noqa: E402
    TrainerProfile,
    algorithm3_pool_formation,
    partition_welfare,
)
from pofl.fl.welfare import WelfareParams  # noqa: E402


def _profiles(rng: np.random.Generator, n: int = 6) -> list[TrainerProfile]:
    out: list[TrainerProfile] = []
    for i in range(n):
        labels = rng.integers(0, 3, size=80)
        out.append(
            TrainerProfile(
                trainer_id=f"t{i}",
                sample_count=80,
                delay=1.0 + 0.05 * i,
                labels=labels,
                num_classes=3,
            )
        )
    return out


def run_sweep() -> None:
    params = WelfareParams(train_time=200.0, psi_local_max=20)
    welfare_alg3: list[float] = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        profs = _profiles(r)
        pools, _ = algorithm3_pool_formation(profs, params, rng=r)
        welfare_alg3.append(partition_welfare(pools, params))
    print("sweep_mean_welfare_algorithm3", statistics.fmean(welfare_alg3))


def spoof_probe() -> None:
    """Spoofing = claiming low loss without training; mitigated by test commitment + holdback."""
    _ = redeem_deposit(loss=0.01, threshold=0.5)
    print("spoof_probe: enforce test_holdback + hash verification")


def sybil_probe() -> None:
    print("sybil_probe: deposits + unique FL submissions per (task, pool, round)")


def main() -> None:
    print("hosting_split_example", split_hosting_fee(100, 3))
    print("slash_stub", slash_fraction(2.0, threshold=1.0))
    run_sweep()
    spoof_probe()
    sybil_probe()


if __name__ == "__main__":
    main()
