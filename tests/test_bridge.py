"""End-to-end integration test: Python -> Substrate pallet-pf-pofl.

Skips automatically unless a dev node is reachable on ws://127.0.0.1:9944.

Bring up the node first:
    docker compose up -d --build node

Then run just this test:
    uv run pytest tests/test_bridge.py -q -s
"""

from __future__ import annotations

import hashlib
import os
import secrets
import socket

import pytest

pytest.importorskip("substrateinterface")

from pofl.chain import extrinsics as ex  # noqa: E402
from pofl.chain.client import ChainClient  # noqa: E402

NODE_URL = os.environ.get("POFL_NODE_URL", "ws://127.0.0.1:9944")


def _node_reachable(url: str) -> bool:
    host_port = url.split("://", 1)[-1].split("/", 1)[0]
    host, _, port = host_port.partition(":")
    try:
        with socket.create_connection((host, int(port or 9944)), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _node_reachable(NODE_URL),
    reason=f"no Substrate node at {NODE_URL} — start one with `docker compose up -d node`",
)


@pytest.fixture(scope="module")
def client() -> ChainClient:
    with ChainClient(NODE_URL) as c:
        yield c


def _addr(kp) -> str:
    return kp.ss58_address


def test_bridge_full_flow(client: ChainClient):
    alice = client.keypair("//Alice")  # requester + sudo
    bob = client.keypair("//Bob")  # pool member / pool wallet
    charlie = client.keypair("//Charlie")  # pool member
    dave = client.keypair("//Dave")  # validator

    # Dev genesis only endows Alice/Bob/AliceStash/BobStash — top up signers.
    client.ensure_funded(alice, _addr(dave), 10**14)

    # --- 1. Build a holdback-compatible test key + commitment.
    test_key = secrets.token_bytes(32)
    commitment = hashlib.blake2b(test_key, digest_size=32).digest()

    head = client.block_number()
    training_deadline = head + 40
    release_block = head + 4

    # --- 2. Requester publishes the task.
    pub = ex.publish_task(
        client,
        alice,
        reward=10_000_000_000_000,
        hosting_fee=2_000_000_000_000,
        participation_deposit=1_000_000_000_000,
        training_deadline=training_deadline,
        release_block=release_block,
        initial_model_cid=b"QmInitialModelCID",
        test_ciphertext_cid=b"QmTestCiphertextCID",
        test_commitment=commitment,
    )
    assert pub.success, f"publish_task failed: {pub.error}"
    published = pub.event("PfPofl", "TaskPublished")
    assert published is not None, "no TaskPublished event"
    task_id = _attr(published, "task_id")
    assert isinstance(task_id, int) and task_id > 0

    # --- 3. Bob locks participation; both pool members are registered.
    lock = ex.lock_participation(client, bob, task_id=task_id)
    assert lock.success, f"lock_participation failed: {lock.error}"

    pool_id = b"pool-alpha"
    members = ex.register_pool_members(
        client, alice, task_id=task_id, pool_id=pool_id,
        members=[_addr(bob), _addr(charlie)],
    )
    assert members.success, f"register_pool_members failed: {members.error}"

    wallet = ex.register_pool_wallet(client, alice, pool_id=pool_id, wallet=_addr(bob))
    assert wallet.success, f"register_pool_wallet failed: {wallet.error}"

    # --- 4. Bob submits an FL model CID for round 1.
    sub = ex.submit_fl_model(
        client, bob, task_id=task_id, pool_id=pool_id, round=1,
        weights_cid=b"QmWeightsRound1",
    )
    assert sub.success, f"submit_fl_model failed: {sub.error}"
    assert sub.event("PfPofl", "FlModelSubmitted") is not None

    # --- 5. Wait past release_block, then reveal the test key.
    client.wait_until_block(release_block + 1)
    reveal = ex.reveal_test_key(client, alice, task_id=task_id, key=test_key)
    assert reveal.success, f"reveal_test_key failed: {reveal.error}"

    # --- 6. Sudo grants Dave validator credit, then Dave scores + votes.
    grant = ex.set_validator_credit(client, alice, who=_addr(dave), credit=100)
    assert grant.success, f"set_validator_credit (sudo) failed: {grant.error}"
    assert client.query("ValidatorCredit", [_addr(dave)]) == 100

    score = ex.report_model_score(
        client, dave, task_id=task_id, pool_id=pool_id, score_q9=1_234_567,
    )
    assert score.success, f"report_model_score failed: {score.error}"

    vote = ex.validator_vote_winner(
        client, dave, task_id=task_id, winner_pool=pool_id,
    )
    assert vote.success, f"validator_vote_winner failed: {vote.error}"

    # --- 7. Requester finalizes; assert TaskSettled event names our pool.
    fin = ex.finalize_task(client, alice, task_id=task_id)
    assert fin.success, f"finalize_task failed: {fin.error}"
    settled = fin.event("PfPofl", "TaskSettled")
    assert settled is not None, "no TaskSettled event"
    winner = _attr(settled, "winner")
    assert _coerce_bytes(winner) == pool_id


def test_bridge_agreement_and_vrf(client: ChainClient):
    """Smoke-test the auxiliary dispatchables: VRF ticket + agreement round."""
    alice = client.keypair("//Alice")
    eve = client.keypair("//Eve")

    client.ensure_funded(alice, _addr(eve), 10**14)

    test_key = secrets.token_bytes(32)
    commitment = hashlib.blake2b(test_key, digest_size=32).digest()
    head = client.block_number()
    pub = ex.publish_task(
        client, alice,
        reward=1_000_000_000_000, hosting_fee=100_000_000_000,
        participation_deposit=10_000_000_000,
        training_deadline=head + 40, release_block=head + 4,
        initial_model_cid=b"QmInit2", test_ciphertext_cid=b"QmCipher2",
        test_commitment=commitment,
    )
    assert pub.success
    task_id = _attr(pub.event("PfPofl", "TaskPublished"), "task_id")

    secret = secrets.token_bytes(32)
    vrf = ex.submit_vrf_ticket(client, eve, task_id=task_id, secret=secret)
    assert vrf.success, f"submit_vrf_ticket failed: {vrf.error}"
    ticket = client.query("VrfTickets", [task_id, _addr(eve)])
    assert isinstance(ticket, int) and ticket > 0

    adv = ex.agreement_advance_round(client, alice, task_id=task_id)
    assert adv.success, f"agreement_advance_round failed: {adv.error}"
    assert client.query("AgreementRound", [task_id]) == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _attr(event: dict, name: str):
    """Pull an attribute out of a normalized event, tolerating list/dict shapes."""
    attrs = event["attributes"]
    if isinstance(attrs, dict):
        return attrs.get(name)
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, dict) and a.get("name") == name:
                return a.get("value")
        # Positional fallback: well-known event orderings.
        order = {
            "TaskPublished": ["task_id", "requester"],
            "TaskSettled": ["task_id", "winner"],
            "FlModelSubmitted": ["task_id", "pool", "round"],
        }
        key = event["event"]
        if key in order and name in order[key]:
            idx = order[key].index(name)
            if idx < len(attrs):
                return attrs[idx]
    return None


def _coerce_bytes(v) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        if v.startswith("0x"):
            return bytes.fromhex(v[2:])
        return v.encode("utf-8")
    if isinstance(v, list) and all(isinstance(b, int) for b in v):
        return bytes(v)
    raise TypeError(f"cannot coerce {type(v).__name__} to bytes")
