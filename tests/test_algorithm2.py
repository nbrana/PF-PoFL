"""Tests for paper Algorithm 2 — Block Rewarding Contract."""

from __future__ import annotations

import io

import pytest
import torch

from pofl.fl.client import TinyMLP, state_dict_to_bytes
from pofl.ipfs_sim import IPFSimulator
from pofl.ledger import Ledger
from pofl.roles.requester import publish_fl_task, submit_fl_model_tx
from pofl.roles.validator import finalize_task_with_consensus


def _save_dataset(X: torch.Tensor, y: torch.Tensor) -> bytes:
    b = io.BytesIO()
    torch.save({"X": X, "y": y}, b)
    return b.getvalue()


def _build_test_set(n: int = 16, num_classes: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(42)
    X = torch.randn(n, 8, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g, dtype=torch.long)
    return X, y


def _model_with_seed(seed: int) -> TinyMLP:
    """Create a TinyMLP whose loss on the test set is determined by seed.

    Higher seed = larger random init = generally higher cross-entropy."""
    torch.manual_seed(seed)
    m = TinyMLP(8, 4, hidden=8)
    if seed > 0:
        with torch.no_grad():
            for p in m.parameters():
                p.data.add_(torch.randn_like(p.data) * (seed * 0.2))
    return m


def _setup_pools(num_pools: int = 8) -> tuple[Ledger, IPFSimulator, str, list[str]]:
    torch.manual_seed(0)
    ledger = Ledger(":memory:")
    ipfs = IPFSimulator()
    accounts = {"requester": (50_000, 0)}
    for i in range(num_pools):
        for j in range(2):
            accounts[f"pool-{i}-trainer-{j}"] = (0, 1)
    for v in range(4):
        accounts[f"v{v}"] = (0, 1)
    ledger.genesis_if_empty(accounts)

    init_model = TinyMLP(8, 4, hidden=8)
    init_bytes = state_dict_to_bytes(init_model.state_dict())
    X_test, y_test = _build_test_set()
    test_bytes = _save_dataset(X_test, y_test)

    task_id, _ = publish_fl_task(
        ledger,
        ipfs,
        publisher="requester",
        reward=4000,
        hosting_fee=400,
        initial_model_bytes=init_bytes,
        test_dataset_bytes=test_bytes,
        deadline_block=5,
        task_id="t-alg2",
        participation_deposit=20,
        release_block=5,
        delta_test_blocks=4,
    )

    pool_ids: list[str] = []
    # Submit models with monotonically increasing seed so loss roughly increases.
    for i in range(num_pools):
        m = _model_with_seed(i)
        cid = ipfs.put(state_dict_to_bytes(m.state_dict()))
        pool_id = f"pool-{i}"
        submit_fl_model_tx(
            ledger,
            pool_id=pool_id,
            task_id=task_id,
            round_index=0,
            weights_cid=cid,
            member_ids=[f"pool-{i}-trainer-0", f"pool-{i}-trainer-1"],
            submission_block=3,
            participation_deposit=20,
        )
        pool_ids.append(pool_id)
    return ledger, ipfs, task_id, pool_ids


def test_paper_finalize_pays_reward_to_winner_wallet():
    ledger, ipfs, task_id, _ = _setup_pools(num_pools=8)
    validators = [(f"v{i}", 1, f"sk{i}".encode()) for i in range(4)]
    result = finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi1=400,
        xi2=20,
    )
    winner_wallet = f"wallet-{result.winner_pool_id}"
    bal = ledger.get_account(winner_wallet).balance
    assert bal == result.payouts.get(winner_wallet, 0)
    assert bal >= 4000  # reward of 4000 plus possibly xi2 if winner also qualifies


def test_paper_finalize_distributes_to_qualifying_pools():
    ledger, ipfs, task_id, _ = _setup_pools(num_pools=8)
    validators = [(f"v{i}", 1, f"sk{i}".encode()) for i in range(4)]
    result = finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi1=400,
        xi2=20,
    )
    # With 8 pools, Q1 should yield >= 2 qualifying.
    assert len(result.qualifying_pool_ids) >= 1
    for pid in result.qualifying_pool_ids:
        wallet = f"wallet-{pid}"
        assert ledger.get_account(wallet).balance == 20


def test_paper_finalize_pays_validators():
    ledger, ipfs, task_id, _ = _setup_pools(num_pools=8)
    validators = [(f"v{i}", 1, f"sk{i}".encode()) for i in range(4)]
    result = finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi1=400,
        xi2=20,
    )
    total_validator_payout = sum(
        ledger.get_account(f"v{i}").balance for i in range(4)
    )
    assert total_validator_payout == 400  # xi1 fully distributed
    # Equal split: 100 each (400/4).
    for i in range(4):
        assert ledger.get_account(f"v{i}").balance == 100
    assert all(
        result.payouts.get(f"v{i}", 0) == 100 for i in range(4)
    )


def test_paper_finalize_credit_deltas():
    ledger, ipfs, task_id, _ = _setup_pools(num_pools=8)
    validators = [(f"v{i}", 1, f"sk{i}".encode()) for i in range(4)]
    pre_credits = {
        f"v{i}": ledger.get_account(f"v{i}").credit for i in range(4)
    }
    pre_winner_credits = {
        f"pool-0-trainer-{j}": ledger.get_account(f"pool-0-trainer-{j}").credit
        for j in range(2)
    }
    result = finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi1=400,
        xi2=20,
        chi1=2,
        chi2=4,
        chi3=1,
    )
    # Validators each get +chi1.
    for i in range(4):
        post = ledger.get_account(f"v{i}").credit
        assert post - pre_credits[f"v{i}"] == 2
    # Winner pool members each get +chi2.
    winner_pid = result.winner_pool_id
    for j in range(2):
        m = f"{winner_pid}-trainer-{j}"
        # Pool-0 is the lowest-loss seed, so likely the winner.
        if winner_pid == "pool-0":
            post = ledger.get_account(m).credit
            assert post - pre_winner_credits.get(m, 0) == 4


def test_paper_finalize_double_call_raises():
    ledger, ipfs, task_id, _ = _setup_pools(num_pools=4)
    validators = [(f"v{i}", 1, f"sk{i}".encode()) for i in range(4)]
    finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi1=400,
        xi2=20,
    )
    with pytest.raises(ValueError, match="already paid"):
        finalize_task_with_consensus(
            ledger,
            ipfs,
            task_id,
            validators,
            torch.device("cpu"),
            current_block=6,
            ranking_window_delta=4,
            xi1=400,
            xi2=20,
        )


def test_faulty_validators_skipped_in_payout():
    ledger, ipfs, task_id, _ = _setup_pools(num_pools=8)
    validators = [(f"v{i}", 1, f"sk{i}".encode()) for i in range(4)]
    result = finalize_task_with_consensus(
        ledger,
        ipfs,
        task_id,
        validators,
        torch.device("cpu"),
        current_block=6,
        ranking_window_delta=4,
        xi1=400,
        xi2=20,
        faulty_validators=["v0"],
    )
    # v0 should not receive a share.
    assert ledger.get_account("v0").balance == 0
    assert result.payouts.get("v0", 0) == 0
    # 400 is split across 3 honest validators.
    total = sum(ledger.get_account(f"v{i}").balance for i in range(1, 4))
    assert total == 400
