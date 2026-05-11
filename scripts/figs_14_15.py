#!/usr/bin/env python3
"""Reproduce paper Figs. 14 and 15 (Sec. VI-D, federation formation game).

Fig. 14: social welfare W(J) for M miners under
  {Algorithm 3, all-singleton, grand-coalition, random}.
Fig. 15: number of pools and number of accepted switch operations
  per iteration of Algorithm 3.

Outputs CSV to stdout. Optional `--plot` produces matplotlib figures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pofl.fl.pool_formation import (  # noqa: E402
    TrainerProfile,
    algorithm3_pool_formation,
)
from pofl.fl.welfare import (  # noqa: E402
    WelfareParams,
    global_label_distribution,
    total_welfare,
)


def make_population(
    n: int, seed: int, *, num_classes: int = 4, samples: int = 60
) -> list[TrainerProfile]:
    rng = np.random.default_rng(seed)
    out: list[TrainerProfile] = []
    for i in range(n):
        bias = rng.integers(0, num_classes)
        logits = rng.standard_normal(num_classes)
        logits[bias] += 2.0
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        labels = rng.choice(num_classes, size=samples, p=p)
        out.append(
            TrainerProfile(
                trainer_id=f"m{i}",
                sample_count=samples,
                delay=1.0 + 0.05 * (i % 5),
                labels=labels,
                num_classes=num_classes,
            )
        )
    return out


def _welfare(partition_indices: Iterable[Iterable[int]],
             profiles: list[TrainerProfile],
             params: WelfareParams) -> float:
    g = global_label_distribution(profiles)
    return total_welfare(
        [[profiles[i] for i in pool] for pool in partition_indices],
        g,
        params,
    )


def scheme_singleton(n: int) -> list[list[int]]:
    return [[i] for i in range(n)]


def scheme_grand_coalition(n: int) -> list[list[int]]:
    return [list(range(n))]


def scheme_random(n: int, *, rng: np.random.Generator) -> list[list[int]]:
    """Random partition with a Dirichlet-like split into ~sqrt(n) pools."""
    k = max(1, int(round(n ** 0.5)))
    assignment = rng.integers(0, k, size=n)
    pools: dict[int, list[int]] = {}
    for i, a in enumerate(assignment):
        pools.setdefault(int(a), []).append(i)
    return [v for v in pools.values() if v]


def run_fig14(out_writer: csv.writer, *, ms: list[int], seeds: list[int],
              params: WelfareParams) -> None:
    out_writer.writerow(["M", "scheme", "seed", "welfare"])
    for M in ms:
        for seed in seeds:
            profs = make_population(M, seed=seed)
            rng = np.random.default_rng(seed + 7919)
            pools_alg3, _ = algorithm3_pool_formation(profs, params, rng=rng)
            idx_pools = [
                [int(profs.index(p)) for p in po.profiles] for po in pools_alg3
            ]
            schemes = {
                "algorithm3": idx_pools,
                "singleton": scheme_singleton(M),
                "grand_coalition": scheme_grand_coalition(M),
                "random": scheme_random(M, rng=np.random.default_rng(seed + 31)),
            }
            for name, partition in schemes.items():
                w = _welfare(partition, profs, params)
                out_writer.writerow([M, name, seed, f"{w:.6f}"])


def run_fig15(out_writer: csv.writer, *, M: int, seed: int,
              params: WelfareParams) -> None:
    out_writer.writerow(["iter", "n_pools", "n_switches", "welfare"])
    profs = make_population(M, seed=seed)
    _, trace = algorithm3_pool_formation(
        profs, params, rng=np.random.default_rng(seed + 7919), trace=True
    )
    for step in trace:
        out_writer.writerow(
            [step.iteration, step.n_pools, step.n_switches, f"{step.total_welfare:.6f}"]
        )


def maybe_plot(fig: int, csv_rows: list[list[str]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        sys.stderr.write("matplotlib not installed; skipping --plot\n")
        return
    if fig == 14:
        # Average welfare per (M, scheme).
        from collections import defaultdict
        agg: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in csv_rows[1:]:
            M, scheme, _seed, w = int(row[0]), row[1], row[2], float(row[3])
            agg[(M, scheme)].append(w)
        ms_sorted = sorted({k[0] for k in agg.keys()})
        schemes = sorted({k[1] for k in agg.keys()})
        plt.figure()
        for s in schemes:
            ys = [
                float(np.mean(agg[(M, s)])) if agg[(M, s)] else 0.0
                for M in ms_sorted
            ]
            plt.plot(ms_sorted, ys, marker="o", label=s)
        plt.xlabel("M (miners)")
        plt.ylabel("Social welfare W(J)")
        plt.legend()
        plt.title("Fig. 14: welfare across schemes")
        out = "fig14.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        sys.stderr.write(f"wrote {out}\n")
    elif fig == 15:
        iters = [int(r[0]) for r in csv_rows[1:]]
        n_pools = [int(r[1]) for r in csv_rows[1:]]
        n_switches = [int(r[2]) for r in csv_rows[1:]]
        fig_, ax1 = plt.subplots()
        ax1.set_xlabel("iteration h")
        ax1.set_ylabel("# pools", color="black")
        ax1.plot(iters, n_pools, marker="*", color="black", label="# pools")
        ax2 = ax1.twinx()
        ax2.set_ylabel("# switch ops", color="blue")
        ax2.plot(iters, n_switches, marker="s", linestyle="--", color="blue", label="# switches")
        plt.title("Fig. 15: partition evolution")
        out = "fig15.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        sys.stderr.write(f"wrote {out}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fig", type=int, choices=[14, 15], required=True)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--seeds", type=int, default=5, help="seeds per M for fig 14")
    args = p.parse_args()

    params = WelfareParams(train_time=200.0, psi_local_max=20)
    rows: list[list[str]] = []

    class _CaptureWriter:
        def writerow(self, row):
            rows.append([str(c) for c in row])

    cw = _CaptureWriter()
    if args.fig == 14:
        run_fig14(
            cw,  # type: ignore[arg-type]
            ms=[10, 20, 40, 80],
            seeds=list(range(args.seeds)),
            params=params,
        )
    else:
        run_fig15(cw, M=20, seed=0, params=params)  # type: ignore[arg-type]

    out = csv.writer(sys.stdout)
    for row in rows:
        out.writerow(row)
    if args.plot:
        maybe_plot(args.fig, rows)


if __name__ == "__main__":
    main()
