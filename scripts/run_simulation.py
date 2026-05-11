#!/usr/bin/env python3
"""End-to-end PF-PoFL simulation: task publish, pools, FL, IPFS CIDs, validators."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOTS = Path(__file__).resolve().parents[1]
if str(ROOTS) not in sys.path:
    sys.path.insert(0, str(ROOTS))

from pofl.data.mnist import (  # noqa: E402
    load_mnist_tensors,
    load_mnist_tensors_4d,
    partition_mnist_tensors,
)
from pofl.fl.client import (  # noqa: E402
    TinyCNN,
    TinyMLP,
    model_from_state_dict,  # noqa: E402
    state_dict_to_bytes,
)
from pofl.fl.pool_formation import (
    TrainerProfile,  # noqa: E402
    algorithm3_pool_formation,  # noqa: E402
)
from pofl.fl.welfare import WelfareParams  # noqa: E402
from pofl.fl.server import evaluate_accuracy, evaluate_loss  # noqa: E402
from pofl.ipfs_sim import IPFSimulator  # noqa: E402
from pofl.ledger import Ledger  # noqa: E402
from pofl.roles.curator import run_pool_federated_rounds  # noqa: E402
from pofl.roles.requester import publish_fl_task, submit_fl_model_tx  # noqa: E402
from pofl.roles.trainer import TrainerNode  # noqa: E402
from pofl.roles.validator import finalize_task_with_consensus  # noqa: E402
from pofl.serialization import decode_model_state_dict_bytes  # noqa: E402


def _save_dataset(X: torch.Tensor, y: torch.Tensor) -> bytes:
    b = io.BytesIO()
    torch.save({"X": X, "y": y}, b)
    return b.getvalue()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["synthetic", "mnist"], default="synthetic")
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-trainers", type=int, default=4)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--clip-norm", type=float, default=2.0)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--sensitivity", type=float, default=2.0)
    p.add_argument("--hidden", type=int, default=24)

    p.add_argument("--samples-per-trainer", type=int, default=120)
    p.add_argument("--test-samples", type=int, default=256)
    p.add_argument("--mnist-root", type=str, default=str((ROOTS / ".data").resolve()))

    p.add_argument("--xi1", type=int, default=500, help="hosting fee distributed to validators")
    p.add_argument("--xi2", type=int, default=0, help="participation deposit + qualifying-pool refund")
    p.add_argument("--ranking-window", type=int, default=8, help="Δ_i blocks for ranking phase")
    p.add_argument("--delta-test-blocks", type=int, default=8)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    print("PF-PoFL simulation config")
    print(f"  task={args.task} seed={args.seed} device={device.type}")
    print(
        "  fl:"
        f" rounds={args.rounds}"
        f" local_epochs={args.local_epochs}"
        f" batch_size={args.batch_size}"
        f" lr={args.lr}"
        f" clip_norm={args.clip_norm}"
        f" sigma={'default' if args.sigma is None else args.sigma}"
        f" sensitivity={args.sensitivity}"
        f" hidden={args.hidden}"
    )
    print(
        "  data:"
        f" num_trainers={args.num_trainers}"
        f" samples_per_trainer={args.samples_per_trainer}"
        f" test_samples={args.test_samples}"
        + (f" mnist_root={args.mnist_root}" if args.task == "mnist" else "")
    )

    if args.task == "mnist":
        input_dim = 28 * 28
        num_classes = 10
        hidden = args.hidden if args.hidden != 24 else 128
        sigma = 0.0 if args.sigma is None else float(args.sigma)

        loader = (
            load_mnist_tensors_4d if args.model == "cnn" else load_mnist_tensors
        )
        Xtr, ytr, Xte, yte = loader(
            data_root=args.mnist_root,
            train_limit=args.num_trainers * args.samples_per_trainer,
            test_limit=args.test_samples,
        )
        parts = partition_mnist_tensors(
            Xtr,
            ytr,
            num_trainers=args.num_trainers,
            samples_per_trainer=args.samples_per_trainer,
            num_classes=num_classes,
            seed=args.seed,
        )
        trainers = []
        for i, part in enumerate(parts):
            pr = TrainerProfile(
                trainer_id=part.trainer_id,
                sample_count=int(part.y.numel()),
                delay=1.0 + 0.1 * i,
                labels=part.labels,
                num_classes=num_classes,
            )
            trainers.append(TrainerNode(trainer_id=part.trainer_id, profile=pr, X=part.X, y=part.y))
        X_test, y_test = Xte, yte
    else:
        input_dim = 8
        num_classes = 4
        hidden = args.hidden
        sigma = 0.02 if args.sigma is None else float(args.sigma)
        rng = np.random.default_rng(0)
        X_test = torch.tensor(rng.standard_normal((args.test_samples, input_dim)), dtype=torch.float32)
        y_test = torch.tensor(rng.integers(0, num_classes, size=(args.test_samples,)), dtype=torch.long)
        trainers = [
            TrainerNode.synthetic_profile(
                f"trainer-{i}",
                rng,
                args.samples_per_trainer,
                num_classes,
                input_dim,
                delay=1.0 + 0.1 * i,
                label_bias=i % num_classes,
            )
            for i in range(args.num_trainers)
        ]

    device = torch.device("cpu")

    ledger = Ledger(":memory:")
    ipfs = IPFSimulator()

    genesis = {
        "requester": (50_000, 10),
        "trainer-0": (0, 5),
        "trainer-1": (0, 5),
        "trainer-2": (0, 5),
        "trainer-3": (0, 5),
        "validator-0": (1000, 20),
        "validator-1": (1000, 18),
        "validator-2": (1000, 22),
        "validator-3": (1000, 16),
    }
    ledger.genesis_if_empty(genesis)

    if args.task == "mnist" and args.model == "cnn":
        model = TinyCNN(num_classes=num_classes)
    else:
        model = TinyMLP(input_dim, num_classes, hidden=hidden)
    init_bytes = state_dict_to_bytes(model.state_dict())

    test_bytes = _save_dataset(X_test, y_test)

    deadline_block = 12

    task_id, task_tx = publish_fl_task(
        ledger,
        ipfs,
        publisher="requester",
        reward=8000,
        hosting_fee=args.xi1,
        initial_model_bytes=init_bytes,
        test_dataset_bytes=test_bytes,
        deadline_block=deadline_block,
        task_id="task-demo",
        participation_deposit=args.xi2,
        release_block=deadline_block,
        delta_test_blocks=args.delta_test_blocks,
    )
    ph = ledger.tip_hash()
    ledger.append_block(
        prev_hash=ph,
        proposer="trainer-0",
        payload={"kind": "include_task", "task_id": task_id},
        tx_hashes=[task_tx],
    )

    welfare_params = WelfareParams(
        train_time=float(args.rounds * 100),
        psi_local_max=max(args.local_epochs * args.rounds, 1),
    )
    pools, _algo3_trace = algorithm3_pool_formation(
        [t.profile for t in trainers],
        welfare_params,
        rng=np.random.default_rng(args.seed),
    )
    data_map = {t.trainer_id: (t.X, t.y) for t in trainers}

    task_row = ledger.get_task(task_id)
    assert task_row is not None
    init_sd_bytes = ipfs.get(task_row.initial_model_cid)
    base_sd = decode_model_state_dict_bytes(init_sd_bytes)

    fl_txs: list[str] = []
    for pool in pools:
        g = model_from_state_dict(base_sd)
        trained = run_pool_federated_rounds(
            pool,
            g,
            data_map,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            clip_norm=args.clip_norm,
            sigma=sigma,
            sensitivity=args.sensitivity,
            device=device,
        )
        wbytes = state_dict_to_bytes(trained.state_dict())
        cid = ipfs.put(wbytes)
        fl_txs.append(
            submit_fl_model_tx(
                ledger,
                pool.pool_id,
                task_id,
                round_index=max(args.rounds - 1, 0),
                weights_cid=cid,
                member_ids=pool.member_ids,
                submission_block=ledger.tip_height(),
                participation_deposit=args.xi2,
            )
        )

    ledger.append_block(
        prev_hash=ledger.tip_hash(),
        proposer="trainer-1",
        payload={"kind": "fl_submissions", "task_id": task_id},
        tx_hashes=fl_txs,
    )

    while ledger.tip_height() < deadline_block:
        ledger.append_block(
            prev_hash=ledger.tip_hash(),
            proposer="synthetic-miner",
            payload={"kind": "empty"},
            tx_hashes=[],
        )

    validators = [
        ("validator-0", ledger.get_account("validator-0").credit, b"sk0-secret"),
        ("validator-1", ledger.get_account("validator-1").credit, b"sk1-secret"),
        ("validator-2", ledger.get_account("validator-2").credit, b"sk2-secret"),
        ("validator-3", ledger.get_account("validator-3").credit, b"sk3-secret"),
    ]

    result = finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        device,
        committee_size=4,
        current_block=ledger.tip_height(),
        ranking_window_delta=args.ranking_window,
        xi1=args.xi1,
        xi2=args.xi2,
    )
    winner = result.winner_pool_id
    height = result.height
    block_hash = result.block_hash
    ba_ok = result.ba_ok
    payee = f"wallet-{winner}"

    # Demo-friendly metrics (same test tensors used for validator ranking).
    ranked = []
    for pool_id, (cid, _r) in ledger.latest_fl_submission_per_pool(task_id).items():
        sd_bytes = ipfs.get(cid)
        sd = decode_model_state_dict_bytes(sd_bytes)
        m = model_from_state_dict(sd)
        ranked.append(
            (
                pool_id,
                evaluate_loss(m, X_test, y_test, device),
                evaluate_accuracy(m, X_test, y_test, device),
            )
        )
    ranked.sort(key=lambda t: t[1])

    print("PF-PoFL simulation complete")
    print(f"  task_id={task_id}")
    print(f"  pools={[p.pool_id for p in pools]}")
    print(f"  winner_pool={winner}")
    for pool_id, loss, acc in ranked:
        print(f"  pool_metrics pool={pool_id} loss={loss:.4f} acc={acc:.4f}")
    print(f"  payout_account={payee} balance={ledger.get_account(payee).balance}")
    print(f"  final_block_height={height} hash={block_hash[:16]}...")
    print(f"  ba_lite_ok={ba_ok}")


if __name__ == "__main__":
    main()
