"""One Python wrapper per `pallet-pf-pofl` dispatchable.

All `BoundedVec<u8, _>` params accept `bytes` or `str` (utf-8 encoded).
All `[u8; 32]` params accept `bytes` of length 32 or a 0x-prefixed hex string.
"""

from __future__ import annotations

from substrateinterface import Keypair

from pofl.chain.client import ChainClient, ExtrinsicResult


def _as_bytes(v: bytes | str) -> bytes:
    if isinstance(v, bytes):
        return v
    return v.encode("utf-8")


def _as_hex32(v: bytes | str) -> str:
    if isinstance(v, str):
        return v if v.startswith("0x") else "0x" + v
    if len(v) != 32:
        raise ValueError(f"expected 32 bytes, got {len(v)}")
    return "0x" + v.hex()


# ---------------------------------------------------------------------------
# Signed extrinsics (any signed origin)
# ---------------------------------------------------------------------------

def publish_task(
    client: ChainClient,
    signer: Keypair,
    *,
    reward: int,
    hosting_fee: int,
    participation_deposit: int,
    training_deadline: int,
    release_block: int,
    initial_model_cid: bytes | str,
    test_ciphertext_cid: bytes | str,
    test_commitment: bytes | str,
) -> ExtrinsicResult:
    """Call 0: publish a task. Escrows reward + hosting_fee + participation_deposit."""
    return client.submit(
        "publish_task",
        {
            "reward": reward,
            "hosting_fee": hosting_fee,
            "participation_deposit": participation_deposit,
            "training_deadline": training_deadline,
            "release_block": release_block,
            "initial_model_cid": _as_bytes(initial_model_cid),
            "test_ciphertext_cid": _as_bytes(test_ciphertext_cid),
            "test_commitment": _as_hex32(test_commitment),
        },
        signer,
    )


def lock_participation(client: ChainClient, signer: Keypair, *, task_id: int) -> ExtrinsicResult:
    """Call 1: lock the per-task participation deposit (ξ_2)."""
    return client.submit("lock_participation", {"task_id": task_id}, signer)


def submit_fl_model(
    client: ChainClient,
    signer: Keypair,
    *,
    task_id: int,
    pool_id: bytes | str,
    round: int,
    weights_cid: bytes | str,
) -> ExtrinsicResult:
    """Call 2: submit a pool's FL aggregate weights CID for a round."""
    return client.submit(
        "submit_fl_model",
        {
            "task_id": task_id,
            "pool_id": _as_bytes(pool_id),
            "round": round,
            "weights_cid": _as_bytes(weights_cid),
        },
        signer,
    )


def reveal_test_key(
    client: ChainClient, signer: Keypair, *, task_id: int, key: bytes | str
) -> ExtrinsicResult:
    """Call 3: reveal the test-key once `release_block` has elapsed."""
    return client.submit(
        "reveal_test_key", {"task_id": task_id, "key": _as_bytes(key)}, signer
    )


def submit_vrf_ticket(
    client: ChainClient, signer: Keypair, *, task_id: int, secret: bytes | str
) -> ExtrinsicResult:
    """Call 4: submit a hash-based VRF lottery ticket for the task committee."""
    return client.submit(
        "submit_vrf_ticket", {"task_id": task_id, "secret": _as_hex32(secret)}, signer
    )


def agreement_advance_round(
    client: ChainClient, signer: Keypair, *, task_id: int
) -> ExtrinsicResult:
    """Call 5: increment the agreement round counter (BBA* stub)."""
    return client.submit("agreement_advance_round", {"task_id": task_id}, signer)


def validator_vote_winner(
    client: ChainClient,
    signer: Keypair,
    *,
    task_id: int,
    winner_pool: bytes | str,
) -> ExtrinsicResult:
    """Call 6: validator vote for the winning pool."""
    return client.submit(
        "validator_vote_winner",
        {"task_id": task_id, "winner_pool": _as_bytes(winner_pool)},
        signer,
    )


def finalize_task(client: ChainClient, signer: Keypair, *, task_id: int) -> ExtrinsicResult:
    """Call 7: requester-signed settlement. Pays winner pool + honest validators."""
    return client.submit("finalize_task", {"task_id": task_id}, signer)


def register_pool_members(
    client: ChainClient,
    signer: Keypair,
    *,
    task_id: int,
    pool_id: bytes | str,
    members: list[str],
) -> ExtrinsicResult:
    """Call 9: register the AccountIds in a pool for the credit accounting."""
    # `BoundedVec<AccountId, _>` is decoded by substrate-interface as a Composite
    # with one inner `Vec<AccountId>` field. Wrapping the list in a single-element
    # list lines up with the Composite's type_mapping length.
    return client.submit(
        "register_pool_members",
        {"task_id": task_id, "pool_id": _as_bytes(pool_id), "members": [members]},
        signer,
    )


def register_pool_wallet(
    client: ChainClient,
    signer: Keypair,
    *,
    pool_id: bytes | str,
    wallet: str,
) -> ExtrinsicResult:
    """Call 10: register the payout wallet for a pool."""
    return client.submit(
        "register_pool_wallet",
        {"pool_id": _as_bytes(pool_id), "wallet": wallet},
        signer,
    )


def report_model_score(
    client: ChainClient,
    signer: Keypair,
    *,
    task_id: int,
    pool_id: bytes | str,
    score_q9: int,
) -> ExtrinsicResult:
    """Call 11: validator reports a pool's model score (loss × 1e9, lower is better)."""
    return client.submit(
        "report_model_score",
        {"task_id": task_id, "pool_id": _as_bytes(pool_id), "score_q9": score_q9},
        signer,
    )


# ---------------------------------------------------------------------------
# Root-origin extrinsics (wrapped via Sudo.sudo)
# ---------------------------------------------------------------------------

def set_validator_credit(
    client: ChainClient,
    sudo_signer: Keypair,
    *,
    who: str,
    credit: int,
) -> ExtrinsicResult:
    """Call 8 (root): set the credit value for a validator account."""
    return client.submit_sudo(
        "set_validator_credit", {"who": who, "credit": credit}, sudo_signer
    )


def mark_faulty(
    client: ChainClient,
    sudo_signer: Keypair,
    *,
    task_id: int,
    validator: str,
) -> ExtrinsicResult:
    """Call 12 (root): mark a validator as faulty for a given task."""
    return client.submit_sudo(
        "mark_faulty", {"task_id": task_id, "validator": validator}, sudo_signer
    )


__all__ = [
    "publish_task",
    "lock_participation",
    "submit_fl_model",
    "reveal_test_key",
    "submit_vrf_ticket",
    "agreement_advance_round",
    "validator_vote_winner",
    "finalize_task",
    "register_pool_members",
    "register_pool_wallet",
    "report_model_score",
    "set_validator_credit",
    "mark_faulty",
]
