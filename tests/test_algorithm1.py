"""Tests for paper Algorithm 1 — Model Ranking Contract."""

from __future__ import annotations

import io

import pytest
import torch

from pofl.fl.client import TinyMLP, state_dict_to_bytes, train_local_sgd
from pofl.ipfs_sim import IPFSimulator
from pofl.ledger import Ledger
from pofl.roles.requester import publish_fl_task, submit_fl_model_tx
from pofl.roles.validator import (
    RankedModel,
    TaskTerminated,
    TooEarly,
    rank_models_for_task,
)


def _save_dataset(X: torch.Tensor, y: torch.Tensor) -> bytes:
    b = io.BytesIO()
    torch.save({"X": X, "y": y}, b)
    return b.getvalue()


def _setup(
    *,
    deadline_block: int = 5,
    delta_test_blocks: int = 4,
    participation_deposit: int = 50,
    publisher_balance: int = 10_000,
) -> tuple[Ledger, IPFSimulator, str, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    ledger = Ledger(":memory:")
    ipfs = IPFSimulator()
    ledger.genesis_if_empty({"requester": (publisher_balance, 0)})
    model = TinyMLP(8, 4, hidden=16)
    init_bytes = state_dict_to_bytes(model.state_dict())
    X_test = torch.randn(32, 8)
    y_test = torch.randint(0, 4, (32,), dtype=torch.long)
    test_bytes = _save_dataset(X_test, y_test)
    task_id, _ = publish_fl_task(
        ledger,
        ipfs,
        publisher="requester",
        reward=1000,
        hosting_fee=100,
        initial_model_bytes=init_bytes,
        test_dataset_bytes=test_bytes,
        deadline_block=deadline_block,
        task_id="t-alg1",
        participation_deposit=participation_deposit,
        release_block=deadline_block,
        delta_test_blocks=delta_test_blocks,
    )
    return ledger, ipfs, task_id, X_test, y_test


def _train_and_submit(
    ledger: Ledger,
    ipfs: IPFSimulator,
    task_id: str,
    pool_id: str,
    *,
    submission_block: int,
    participation_deposit: int,
    epochs: int = 1,
) -> str:
    torch.manual_seed(hash(pool_id) & 0xFFFF)
    m = TinyMLP(8, 4, hidden=16)
    X = torch.randn(64, 8)
    y = torch.randint(0, 4, (64,), dtype=torch.long)
    train_local_sgd(m, X, y, epochs=epochs, batch_size=16, lr=0.05, device=torch.device("cpu"))
    cid = ipfs.put(state_dict_to_bytes(m.state_dict()))
    return submit_fl_model_tx(
        ledger,
        pool_id=pool_id,
        task_id=task_id,
        round_index=0,
        weights_cid=cid,
        member_ids=[f"{pool_id}-trainer-0"],
        submission_block=submission_block,
        participation_deposit=participation_deposit,
    )


def test_too_early_raises_before_deadline():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=10)
    _train_and_submit(
        ledger, ipfs, task_id, "pool-a", submission_block=2, participation_deposit=50
    )
    with pytest.raises(TooEarly):
        rank_models_for_task(
            ledger,
            ipfs,
            task_id,
            torch.device("cpu"),
            current_block=4,
            ranking_window_delta=4,
        )


def test_too_late_raises_past_window():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=5, delta_test_blocks=2)
    _train_and_submit(
        ledger, ipfs, task_id, "pool-a", submission_block=2, participation_deposit=50
    )
    with pytest.raises(TaskTerminated):
        rank_models_for_task(
            ledger,
            ipfs,
            task_id,
            torch.device("cpu"),
            current_block=10,  # > 5 + 2
            ranking_window_delta=2,
        )


def test_late_submission_marked_late():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=5)
    _train_and_submit(
        ledger,
        ipfs,
        task_id,
        "pool-late",
        submission_block=99,  # past deadline
        participation_deposit=50,
    )
    ranked = rank_models_for_task(
        ledger,
        ipfs,
        task_id,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi2=50,
    )
    assert len(ranked) == 1
    assert ranked[0].status == "late"
    assert ranked[0].score is None


def test_insufficient_deposit_marked_late():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=5)
    _train_and_submit(
        ledger,
        ipfs,
        task_id,
        "pool-cheap",
        submission_block=3,
        participation_deposit=10,  # < xi2=50
    )
    ranked = rank_models_for_task(
        ledger,
        ipfs,
        task_id,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi2=50,
    )
    assert len(ranked) == 1
    assert ranked[0].status == "late"


def test_corrupt_blob_marked_unevaluated():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=5)
    _train_and_submit(
        ledger, ipfs, task_id, "pool-a", submission_block=3, participation_deposit=50
    )
    # Corrupt only the weights blob (not the test dataset).
    weights_cid, _round = ledger.latest_fl_submission_per_pool(task_id)["pool-a"]
    ipfs._mem[weights_cid] = b"\x00\x01\x02not-a-state-dict"  # type: ignore[attr-defined]
    ranked = rank_models_for_task(
        ledger,
        ipfs,
        task_id,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi2=50,
    )
    statuses = {r.status for r in ranked}
    assert "unevaluated" in statuses


def test_happy_path_evaluated_and_ordered():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=5)
    _train_and_submit(
        ledger, ipfs, task_id, "pool-a", submission_block=3, participation_deposit=50
    )
    _train_and_submit(
        ledger, ipfs, task_id, "pool-b", submission_block=3, participation_deposit=50
    )
    ranked = rank_models_for_task(
        ledger,
        ipfs,
        task_id,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi2=50,
    )
    assert len(ranked) == 2
    for r in ranked:
        assert r.status == "evaluated"
        assert r.score is not None
    # Ascending by loss.
    assert ranked[0].score <= ranked[1].score


def test_evaluated_sorts_before_late_or_unevaluated():
    ledger, ipfs, task_id, _, _ = _setup(deadline_block=5)
    _train_and_submit(
        ledger, ipfs, task_id, "good", submission_block=3, participation_deposit=50
    )
    _train_and_submit(
        ledger, ipfs, task_id, "late", submission_block=99, participation_deposit=50
    )
    ranked = rank_models_for_task(
        ledger,
        ipfs,
        task_id,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi2=50,
    )
    assert ranked[0].status == "evaluated"
    assert ranked[-1].status == "late"
    assert isinstance(ranked[0], RankedModel)
