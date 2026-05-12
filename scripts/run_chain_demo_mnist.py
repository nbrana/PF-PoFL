#!/usr/bin/env python3
"""MNIST federated-learning demo against a live `pallet-pf-pofl` Substrate node.

Each pool trains a TinyMLP on real MNIST data, stores the weights in a local
content-addressed store (sha256), submits the on-chain CID via the bridge, the
validator computes a real test-set loss for each pool and scores/votes on
chain, and after finalize the winning weights are pulled back and evaluated
for accuracy.

    docker compose up -d node
    uv run python scripts/run_chain_demo_mnist.py
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pofl.chain import extrinsics as ex  # noqa: E402
from pofl.chain.client import ChainClient, ExtrinsicResult  # noqa: E402
from pofl.data.mnist import load_mnist_tensors  # noqa: E402
from pofl.fl.client import (  # noqa: E402
    TinyMLP,
    model_from_state_dict,
    state_dict_from_bytes,
    state_dict_to_bytes,
    train_local_sgd,
)
from pofl.fl.server import evaluate_accuracy, evaluate_loss  # noqa: E402

REWARD = 10_000_000_000_000
HOSTING_FEE = 2_000_000_000_000
PARTICIPATION_DEPOSIT = 1_000_000_000_000

POOL_A = b"pool-mnist-A"
POOL_B = b"pool-mnist-B"


# ---------------------------------------------------------------------------
# Local content-addressed store (stand-in for IPFS, keyed by raw sha256 digest
# so the 32-byte CID fits the pallet's 64-byte BoundedVec bound).
# ---------------------------------------------------------------------------

_LOCAL_IPFS: dict[bytes, bytes] = {}


def ipfs_put(content: bytes) -> bytes:
    cid = hashlib.sha256(content).digest()
    _LOCAL_IPFS[cid] = content
    return cid


def ipfs_get(cid: bytes) -> bytes:
    return _LOCAL_IPFS[cid]


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def show(label: str, res: ExtrinsicResult) -> None:
    status = "ok" if res.success else f"FAIL ({res.error})"
    print(f"  [{label:<26}] {status}")
    for ev in res.events:
        if ev["pallet"] == "PfPofl":
            print(f"      event PfPofl.{ev['event']}")
    if not res.success:
        sys.exit(1)


def coerce_bytes(v) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return bytes.fromhex(v[2:]) if v.startswith("0x") else v.encode()
    if isinstance(v, list) and all(isinstance(b, int) for b in v):
        return bytes(v)
    raise TypeError(f"cannot coerce {type(v).__name__} to bytes")


def attr(event, name: str):
    if event is None:
        return None
    attrs = event["attributes"]
    if isinstance(attrs, dict):
        return attrs.get(name)
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, dict) and a.get("name") == name:
                return a.get("value")
        positional = {
            "TaskPublished": ["task_id", "requester"],
            "TaskSettled": ["task_id", "winner"],
        }
        if event["event"] in positional and name in positional[event["event"]]:
            return attrs[positional[event["event"]].index(name)]
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pool(name: str, X: torch.Tensor, y: torch.Tensor, hidden: int, epochs: int, lr: float):
    model = TinyMLP(input_dim=X.shape[1], num_classes=10, hidden=hidden)
    t0 = time.time()
    train_local_sgd(model, X, y, epochs=epochs, batch_size=32, lr=lr, device=torch.device("cpu"))
    print(f"  trained {name:<8} on {X.shape[0]:>4} samples in {time.time() - t0:0.2f}s")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:9944")
    ap.add_argument("--mnist-root", default=".data")
    ap.add_argument("--pool-a-samples", type=int, default=600)
    ap.add_argument("--pool-b-samples", type=int, default=120)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    banner("Load MNIST")
    Xtr, ytr, Xte, yte = load_mnist_tensors(
        data_root=args.mnist_root,
        train_limit=args.pool_a_samples + args.pool_b_samples,
        test_limit=args.test_samples,
    )
    print(f"  train shape {tuple(Xtr.shape)}, test shape {tuple(Xte.shape)}")

    Xa, ya = Xtr[: args.pool_a_samples], ytr[: args.pool_a_samples]
    Xb = Xtr[args.pool_a_samples : args.pool_a_samples + args.pool_b_samples]
    yb = ytr[args.pool_a_samples : args.pool_a_samples + args.pool_b_samples]

    banner("Train per-pool models (FedAvg over 1 trainer per pool, for brevity)")
    model_a = train_pool("pool-A", Xa, ya, args.hidden, args.epochs, args.lr)
    model_b = train_pool("pool-B", Xb, yb, args.hidden, args.epochs, args.lr)

    banner("Store weights in local content-addressed IPFS-like store")
    bytes_a = state_dict_to_bytes(model_a.state_dict())
    bytes_b = state_dict_to_bytes(model_b.state_dict())
    cid_a = ipfs_put(bytes_a)
    cid_b = ipfs_put(bytes_b)
    print(f"  pool-A cid = {cid_a.hex()[:20]}... ({len(bytes_a)} bytes)")
    print(f"  pool-B cid = {cid_b.hex()[:20]}... ({len(bytes_b)} bytes)")

    # Validator pre-scores both pools (loss; lower is better). On-chain pallet
    # only stores u64 score_q9 (loss × 1e9). Mirror that quantization here.
    banner("Validator: evaluate each pool against the (revealed) MNIST test set")
    device = torch.device("cpu")
    loss_a = evaluate_loss(model_a, Xte, yte, device)
    loss_b = evaluate_loss(model_b, Xte, yte, device)
    score_a_q9 = int(loss_a * 1e9)
    score_b_q9 = int(loss_b * 1e9)
    print(f"  pool-A loss = {loss_a:0.4f}   score_q9 = {score_a_q9}")
    print(f"  pool-B loss = {loss_b:0.4f}   score_q9 = {score_b_q9}")
    expected_winner_pool = POOL_A if score_a_q9 <= score_b_q9 else POOL_B
    print(f"  expected winner pool = {expected_winner_pool!r}")

    # -----------------------------------------------------------------
    # Chain flow
    # -----------------------------------------------------------------

    with ChainClient(args.url) as c:
        alice = c.keypair("//Alice")  # requester + sudo
        bob = c.keypair("//Bob")  # pool-A captain + payout wallet
        astash = c.keypair("//Alice//stash")  # pool-B captain + payout wallet
        bstash = c.keypair("//Bob//stash")  # validator

        print(f"\nConnected to {args.url}  (head block {c.block_number()})")
        print(f"  Alice         = {alice.ss58_address}")
        print(f"  Bob           = {bob.ss58_address}      (pool-A)")
        print(f"  Alice//stash  = {astash.ss58_address}  (pool-B)")
        print(f"  Bob//stash    = {bstash.ss58_address}  (validator)")

        # Initial model CID (just hash the freshly-initialised model bytes).
        initial_bytes = state_dict_to_bytes(
            TinyMLP(input_dim=784, num_classes=10, hidden=args.hidden).state_dict()
        )
        initial_cid = ipfs_put(initial_bytes)

        # Holdback test-key commitment.
        test_key = secrets.token_bytes(32)
        commitment = hashlib.blake2b(test_key, digest_size=32).digest()
        test_ciphertext_cid = ipfs_put(b"opaque-encrypted-mnist-testset-stub")

        head = c.block_number()
        training_deadline = head + 40
        release_block = head + 4

        banner("1. publish_task")
        pub = ex.publish_task(
            c, alice,
            reward=REWARD, hosting_fee=HOSTING_FEE,
            participation_deposit=PARTICIPATION_DEPOSIT,
            training_deadline=training_deadline,
            release_block=release_block,
            initial_model_cid=initial_cid,
            test_ciphertext_cid=test_ciphertext_cid,
            test_commitment=commitment,
        )
        show("publish_task", pub)
        task_id = attr(pub.event("PfPofl", "TaskPublished"), "task_id")
        print(f"  -> task_id = {task_id}")

        banner("2. Pool captains lock participation")
        show("lock_participation (A)", ex.lock_participation(c, bob, task_id=task_id))
        show("lock_participation (B)", ex.lock_participation(c, astash, task_id=task_id))

        banner("3. Register both pools' members + payout wallets")
        for pool_id, captain in [(POOL_A, bob), (POOL_B, astash)]:
            show(f"register_pool_members ({pool_id.decode()})",
                 ex.register_pool_members(c, alice, task_id=task_id, pool_id=pool_id,
                                          members=[captain.ss58_address]))
            show(f"register_pool_wallet ({pool_id.decode()})",
                 ex.register_pool_wallet(c, alice, pool_id=pool_id, wallet=captain.ss58_address))

        banner("4. Each pool submits its trained model CID")
        show("submit_fl_model (A)",
             ex.submit_fl_model(c, bob, task_id=task_id, pool_id=POOL_A,
                                round=1, weights_cid=cid_a))
        show("submit_fl_model (B)",
             ex.submit_fl_model(c, astash, task_id=task_id, pool_id=POOL_B,
                                round=1, weights_cid=cid_b))

        banner(f"5. Wait until block {release_block + 1}, then reveal test key")
        c.wait_until_block(release_block + 1)
        show("reveal_test_key", ex.reveal_test_key(c, alice, task_id=task_id, key=test_key))

        banner("6. Validator (Bob//stash) is granted credit, scores both pools, votes")
        show("set_validator_credit (sudo)",
             ex.set_validator_credit(c, alice, who=bstash.ss58_address, credit=100))
        show("report_model_score (A)",
             ex.report_model_score(c, bstash, task_id=task_id, pool_id=POOL_A, score_q9=score_a_q9))
        show("report_model_score (B)",
             ex.report_model_score(c, bstash, task_id=task_id, pool_id=POOL_B, score_q9=score_b_q9))
        show("validator_vote_winner",
             ex.validator_vote_winner(c, bstash, task_id=task_id, winner_pool=expected_winner_pool))

        banner("7. finalize_task")
        winner_wallet = bob.ss58_address if expected_winner_pool == POOL_A else astash.ss58_address
        before = c.free_balance(winner_wallet)
        fin = ex.finalize_task(c, alice, task_id=task_id)
        show("finalize_task", fin)
        gained = c.free_balance(winner_wallet) - before
        on_chain_winner = coerce_bytes(attr(fin.event("PfPofl", "TaskSettled"), "winner"))
        print(f"  on-chain winner pool = {on_chain_winner!r}")
        print(f"  winner pool wallet gained: {gained / 1e12:0.4f} UNIT")

        banner("8. Pull winning model from chain → IPFS → evaluate accuracy")
        sub = c.query("FlSubmissions", [task_id, on_chain_winner])
        if sub is None:
            print("  FlSubmissions returned None — cannot evaluate")
            return 1
        # NMap value is (round, weights_cid). Tolerate both tuple- and dict-shapes.
        if isinstance(sub, (list, tuple)) and len(sub) == 2:
            round_n, weights_cid_v = sub
        elif isinstance(sub, dict):
            round_n = sub.get("col1") or sub.get(0)
            weights_cid_v = sub.get("col2") or sub.get(1)
        else:
            print(f"  unexpected FlSubmissions shape: {sub!r}")
            return 1

        weights_cid = coerce_bytes(weights_cid_v)
        weights = ipfs_get(weights_cid)
        winning_model = model_from_state_dict(state_dict_from_bytes(weights))
        acc = evaluate_accuracy(winning_model, Xte, yte, device)
        print(f"  on-chain round = {round_n}")
        print(f"  fetched weights ({len(weights)} bytes) for cid {weights_cid.hex()[:20]}...")
        print(f"  WINNING MODEL TEST ACCURACY = {acc * 100:0.2f}%  (on {Xte.shape[0]} samples)")
        # Cross-check against the losing pool for context.
        losing_pool = POOL_B if expected_winner_pool == POOL_A else POOL_A
        losing_cid = cid_b if expected_winner_pool == POOL_A else cid_a
        losing_model = model_from_state_dict(state_dict_from_bytes(ipfs_get(losing_cid)))
        losing_acc = evaluate_accuracy(losing_model, Xte, yte, device)
        print(f"  (losing pool {losing_pool!r} accuracy = {losing_acc * 100:0.2f}%)")

        banner("done")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
