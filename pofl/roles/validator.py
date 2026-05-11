from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch

from pofl.consensus import ba as ba_mod
from pofl.economics import quartile_threshold
from pofl.fl.client import model_from_state_dict
from pofl.fl.server import evaluate_loss
from pofl.ipfs_sim import IPFSimulator
from pofl.ledger import Ledger
from pofl.serialization import decode_model_state_dict_bytes, decode_test_dataset_bytes
from pofl.types import FLModelPayload


RankStatus = Literal["evaluated", "unevaluated", "late", "early"]


class TooEarly(Exception):
    """Raised when current block height is before the task's training deadline."""


class TaskTerminated(Exception):
    """Raised when current block is past the testing window i_τ + Δ_i."""


@dataclass
class RankedModel:
    pool_id: str
    weights_cid: str
    score: Optional[float]
    status: RankStatus


@dataclass
class FinalizationResult:
    winner_pool_id: str
    qualifying_pool_ids: list[str]
    ranking: list[RankedModel]
    payouts: dict[str, int]
    credit_deltas: dict[str, int]
    height: int
    block_hash: str
    ba_ok: bool
    threshold: float = 0.0


def verify_test_dataset(ipfs: IPFSimulator, test_cid: str, expected_hash: str) -> bytes:
    blob = ipfs.get(test_cid)
    h = hashlib.sha256(blob).hexdigest()
    if h != expected_hash:
        raise ValueError("test dataset hash mismatch on chain commitment")
    return blob


def rank_models_for_task(
    ledger: Ledger,
    ipfs: IPFSimulator,
    task_id: str,
    device: torch.device,
    *,
    current_block: int,
    ranking_window_delta: int,
    xi2: int = 0,
) -> list[RankedModel]:
    """Paper Algorithm 1 — Model Ranking Contract.

    Phase gates:
      current_block < deadline_block             -> raises TooEarly
      current_block > deadline_block + delta     -> raises TaskTerminated
      otherwise                                  -> evaluate models in window

    Per-submission gates:
      participation_deposit < xi2 OR submission_block > deadline_block
        -> RankedModel(status="late", score=None)
      decode/IPFS error
        -> RankedModel(status="unevaluated", score=None)

    Sort: evaluated (ascending loss) before late/unevaluated, stable on pool_id.
    """
    task = ledger.get_task(task_id)
    if task is None:
        raise ValueError("unknown task")
    if task.escrow_amount <= 0:
        raise ValueError("task has no escrow")

    if current_block < task.deadline_block:
        raise TooEarly(
            f"current_block={current_block} < deadline_block={task.deadline_block}"
        )
    if current_block > task.deadline_block + ranking_window_delta:
        raise TaskTerminated(
            f"current_block={current_block} past testing window "
            f"({task.deadline_block} + {ranking_window_delta})"
        )

    test_blob = verify_test_dataset(ipfs, task.test_dataset_cid, task.test_dataset_hash)
    X, y = decode_test_dataset_bytes(test_blob, device=device)

    latest = ledger.latest_fl_submission_per_pool(task_id)
    out: list[RankedModel] = []
    for pool_id, (cid, _round) in latest.items():
        meta = ledger.get_fl_submission_meta(task_id, pool_id)
        if meta is None:
            out.append(RankedModel(pool_id, cid, None, "unevaluated"))
            continue
        sub_block, deposit, weights_cid = meta
        if deposit < xi2 or sub_block > task.deadline_block:
            out.append(RankedModel(pool_id, weights_cid, None, "late"))
            continue
        try:
            sd_bytes = ipfs.get(weights_cid)
            sd = decode_model_state_dict_bytes(sd_bytes)
            model = model_from_state_dict(sd)
            loss = evaluate_loss(model, X, y, device)
        except (ValueError, KeyError, RuntimeError):
            out.append(RankedModel(pool_id, weights_cid, None, "unevaluated"))
            continue
        out.append(RankedModel(pool_id, weights_cid, float(loss), "evaluated"))

    def sort_key(rm: RankedModel) -> tuple[int, float, str]:
        rank_bucket = 0 if rm.status == "evaluated" else 1
        score = rm.score if rm.score is not None else float("inf")
        return (rank_bucket, score, rm.pool_id)

    out.sort(key=sort_key)
    return out


def _winning_pool_members(
    ledger: Ledger, task_id: str, pool_id: str
) -> list[str]:
    """Recover the member list from the FL-model transaction payload."""
    with ledger._conn:  # type: ignore[attr-defined]
        rows = ledger._conn.execute(  # type: ignore[attr-defined]
            """SELECT t.payload_json FROM transactions t
               JOIN fl_submissions f ON f.tx_hash = t.tx_hash
               WHERE f.task_id = ? AND f.pool_id = ?
               ORDER BY f.round_index DESC LIMIT 1""",
            (task_id, pool_id),
        ).fetchall()
    if not rows:
        return []
    payload = FLModelPayload.from_json(str(rows[0][0]))
    try:
        return list(json.loads(payload.member_ids_json))
    except (TypeError, ValueError):
        return []


def finalize_task_with_consensus(
    ledger: Ledger,
    ipfs: IPFSimulator,
    task_id: str,
    validators: list[tuple[str, int, bytes]],
    device: torch.device,
    committee_size: int = 4,
    *,
    current_block: Optional[int] = None,
    ranking_window_delta: int = 0,
    xi1: int = 0,
    xi2: int = 0,
    chi1: int = 2,
    chi2: int = 4,
    chi3: int = 1,
    faulty_validators: Optional[Iterable[str]] = None,
) -> FinalizationResult:
    """Paper Algorithm 2 — Block Rewarding Contract.

    1. Run Algorithm 1 to rank models.
    2. Build W_τ = pools with loss ≤ first-quartile threshold.
    3. Winner = best-ranked in W_τ.
    4. Pay reward to winner pool wallet, hosting fee + redistributed ξ₂ to
       honest validators, ξ₂ to qualifying-non-winner pool wallets.
    5. Apply credit deltas χ₁/χ₂/χ₃.
    6. Append finalization block.
    """
    task = ledger.get_task(task_id)
    if task is None:
        raise ValueError("task missing")
    if current_block is None:
        current_block = task.deadline_block

    ranked = rank_models_for_task(
        ledger,
        ipfs,
        task_id,
        device,
        current_block=current_block,
        ranking_window_delta=ranking_window_delta,
        xi2=xi2,
    )
    evaluated = [r for r in ranked if r.status == "evaluated" and r.score is not None]
    if not evaluated:
        raise ValueError("no evaluated model submissions to finalize")

    losses = [r.score for r in evaluated if r.score is not None]
    threshold = quartile_threshold(losses)
    qualifying = [r for r in evaluated if r.score is not None and r.score <= threshold]
    if not qualifying:
        qualifying = [evaluated[0]]
    winner = qualifying[0]
    winner_pool_id = winner.pool_id
    qualifying_non_winner = [r for r in qualifying if r.pool_id != winner_pool_id]

    # Pool wallets.
    winner_wallet = ledger.upsert_pool_wallet(
        winner_pool_id, f"wallet-{winner_pool_id}"
    )
    qualifying_wallets = [
        ledger.upsert_pool_wallet(r.pool_id, f"wallet-{r.pool_id}")
        for r in qualifying_non_winner
    ]

    # Validators: register all and apply faulty marks.
    faulty_set = set(faulty_validators or [])
    for vid, _credit, _sk in validators:
        ledger.register_validator(task_id, vid)
        if vid in faulty_set:
            ledger.mark_validator_faulty(task_id, vid)
    honest = ledger.honest_validators(task_id)

    # Credit deltas.
    miner_credit_deltas: list[tuple[str, int]] = []
    winner_members = _winning_pool_members(ledger, task_id, winner_pool_id)
    for member in winner_members:
        miner_credit_deltas.append((member, chi2))
    for r in qualifying_non_winner:
        for member in _winning_pool_members(ledger, task_id, r.pool_id):
            miner_credit_deltas.append((member, chi3))

    validator_credit_deltas = [(vid, chi1) for vid in honest]

    payouts = ledger.finalize_task_paper(
        task_id=task_id,
        winner_pool_id=winner_pool_id,
        winner_pool_wallet=winner_wallet,
        qualifying_pool_wallets=qualifying_wallets,
        honest_validator_ids=honest,
        xi1=xi1,
        xi2=xi2,
        miner_credit_deltas=miner_credit_deltas,
        validator_credit_deltas=validator_credit_deltas,
    )

    seed = ledger.tip_hash()
    committee = ba_mod.select_committee_weighted_vrf(validators, seed, committee_size)
    payload = {
        "kind": "finalize_task",
        "task_id": task_id,
        "winner_pool": winner_pool_id,
        "qualifying_pools": [r.pool_id for r in qualifying_non_winner],
        "ranking": [
            {"pool": r.pool_id, "loss": r.score, "status": r.status} for r in ranked
        ],
        "committee": committee,
        "threshold": threshold,
    }

    candidate_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    ba_ok = ba_mod.ba_lite_finalize(committee, candidate_hash, prev_seed=seed)
    if not ba_ok:
        payload["ba_ok"] = False
    height, block_hash = ledger.append_block(
        prev_hash=seed,
        proposer=committee[0] if committee else "no_committee",
        payload=payload,
        tx_hashes=[],
    )

    credit_deltas: dict[str, int] = {}
    for acct, d in miner_credit_deltas:
        credit_deltas[acct] = credit_deltas.get(acct, 0) + d
    for vid, d in validator_credit_deltas:
        credit_deltas[vid] = credit_deltas.get(vid, 0) + d

    return FinalizationResult(
        winner_pool_id=winner_pool_id,
        qualifying_pool_ids=[r.pool_id for r in qualifying_non_winner],
        ranking=ranked,
        payouts=payouts,
        credit_deltas=credit_deltas,
        height=height,
        block_hash=block_hash,
        ba_ok=ba_ok,
        threshold=float(threshold) if threshold != float("inf") else 0.0,
    )
