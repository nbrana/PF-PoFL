#!/usr/bin/env python3
"""End-to-end demo against a live `pallet-pf-pofl` Substrate node.

Run a dev node first:
    docker compose up -d --build node

Then:
    uv run python scripts/run_chain_demo.py

Override the WS endpoint with --url ws://host:9944 if needed.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pofl.chain import extrinsics as ex  # noqa: E402
from pofl.chain.client import ChainClient, ExtrinsicResult  # noqa: E402

REWARD = 10_000_000_000_000
HOSTING_FEE = 2_000_000_000_000
PARTICIPATION_DEPOSIT = 1_000_000_000_000
POOL_ID = b"pool-alpha"
INITIAL_MODEL_CID = b"QmInitialModelCID-demo"
TEST_CIPHERTEXT_CID = b"QmTestCiphertextCID-demo"
WEIGHTS_CID = b"QmWeightsRound1-demo"


def banner(s: str) -> None:
    print(f"\n=== {s} ===")


def show(label: str, res: ExtrinsicResult) -> None:
    status = "ok" if res.success else f"FAIL ({res.error})"
    print(f"  [{label:<22}] {status}  block={res.block_hash[:18]}...")
    for ev in res.events:
        if ev["pallet"] == "PfPofl":
            print(f"      event PfPofl.{ev['event']}  attrs={ev['attributes']}")
    if not res.success:
        sys.exit(1)


def fmt_balance(n: int) -> str:
    return f"{n / 1e12:.4f} UNIT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:9944")
    args = ap.parse_args()

    with ChainClient(args.url) as c:
        alice = c.keypair("//Alice")
        bob = c.keypair("//Bob")
        charlie = c.keypair("//Charlie")
        dave = c.keypair("//Dave")

        print(f"Connected to {args.url}")
        print(f"  chain head block = {c.block_number()}")
        print(f"  Alice   = {alice.ss58_address}")
        print(f"  Bob     = {bob.ss58_address}  (pool member + pool wallet)")
        print(f"  Charlie = {charlie.ss58_address}  (pool member)")
        print(f"  Dave    = {dave.ss58_address}  (validator)")

        banner("0. Pre-fund non-endowed signers")
        c.ensure_funded(alice, dave.ss58_address, 10**14)
        print(f"  Dave balance: {fmt_balance(c.free_balance(dave.ss58_address))}")

        banner("1. Requester (Alice) publishes a task")
        test_key = secrets.token_bytes(32)
        commitment = hashlib.blake2b(test_key, digest_size=32).digest()
        head = c.block_number()
        training_deadline = head + 40
        release_block = head + 4

        alice_before = c.free_balance(alice.ss58_address)
        pub = ex.publish_task(
            c, alice,
            reward=REWARD, hosting_fee=HOSTING_FEE,
            participation_deposit=PARTICIPATION_DEPOSIT,
            training_deadline=training_deadline,
            release_block=release_block,
            initial_model_cid=INITIAL_MODEL_CID,
            test_ciphertext_cid=TEST_CIPHERTEXT_CID,
            test_commitment=commitment,
        )
        show("publish_task", pub)
        task_id = _attr(pub.event("PfPofl", "TaskPublished"), "task_id")
        print(f"  -> task_id = {task_id}")
        print(f"  -> training_deadline = {training_deadline}, release_block = {release_block}")
        alice_after = c.free_balance(alice.ss58_address)
        print(f"  Alice escrowed: {fmt_balance(alice_before - alice_after)}")

        banner("2. Bob locks his participation deposit")
        bob_before = c.free_balance(bob.ss58_address)
        show("lock_participation", ex.lock_participation(c, bob, task_id=task_id))
        print(f"  Bob deposit: {fmt_balance(bob_before - c.free_balance(bob.ss58_address))}")

        banner("3. Register pool members + payout wallet")
        show("register_pool_members",
             ex.register_pool_members(c, alice, task_id=task_id, pool_id=POOL_ID,
                                      members=[bob.ss58_address, charlie.ss58_address]))
        show("register_pool_wallet",
             ex.register_pool_wallet(c, alice, pool_id=POOL_ID, wallet=bob.ss58_address))

        banner("4. Pool submits its aggregated FL model CID")
        show("submit_fl_model",
             ex.submit_fl_model(c, bob, task_id=task_id, pool_id=POOL_ID,
                                round=1, weights_cid=WEIGHTS_CID))

        banner(f"5. Wait for release_block={release_block} then reveal test key")
        c.wait_until_block(release_block + 1)
        print(f"  chain head now = {c.block_number()}")
        show("reveal_test_key",
             ex.reveal_test_key(c, alice, task_id=task_id, key=test_key))

        banner("6. Sudo grants Dave validator credit, Dave scores + votes")
        show("set_validator_credit (sudo)",
             ex.set_validator_credit(c, alice, who=dave.ss58_address, credit=100))
        print(f"  ValidatorCredit[Dave] = {c.query('ValidatorCredit', [dave.ss58_address])}")
        show("report_model_score",
             ex.report_model_score(c, dave, task_id=task_id, pool_id=POOL_ID, score_q9=1_234_567))
        show("validator_vote_winner",
             ex.validator_vote_winner(c, dave, task_id=task_id, winner_pool=POOL_ID))

        banner("7. Requester finalizes; winner pool gets paid")
        winner_wallet = bob.ss58_address
        winner_before = c.free_balance(winner_wallet)
        dave_before = c.free_balance(dave.ss58_address)
        show("finalize_task", ex.finalize_task(c, alice, task_id=task_id))
        winner_after = c.free_balance(winner_wallet)
        dave_after = c.free_balance(dave.ss58_address)
        print(f"  winner pool wallet (Bob) gained: {fmt_balance(winner_after - winner_before)}")
        print(f"  validator Dave gained:           {fmt_balance(dave_after - dave_before)}")
        print(f"  MinerCredit[Bob]     = {c.query('MinerCredit', [bob.ss58_address])}")
        print(f"  MinerCredit[Charlie] = {c.query('MinerCredit', [charlie.ss58_address])}")
        print(f"  ValidatorCredit[Dave]= {c.query('ValidatorCredit', [dave.ss58_address])}")

        banner("done")
        return 0


def _attr(event, name: str):
    if event is None:
        return None
    attrs = event["attributes"]
    if isinstance(attrs, dict):
        return attrs.get(name)
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, dict) and a.get("name") == name:
                return a.get("value")
        order = {"TaskPublished": ["task_id", "requester"]}
        key = event["event"]
        if key in order and name in order[key]:
            return attrs[order[key].index(name)]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
