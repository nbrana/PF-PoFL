import numpy as np
import pytest

from pofl.consensus.ba import multi_round_graded_consensus
from pofl.economics import split_hosting_fee
from pofl.fl.emd import emd_histograms
from pofl.fl.pool_formation import TrainerProfile, algorithm3_pool_formation
from pofl.fl.welfare import WelfareParams
from pofl.ipfs_sim import IPFSimulator
from pofl.ledger import Ledger
from pofl.roles.requester import publish_fl_task
from pofl.test_holdback import decrypt_test, prepare_test_holdback, reveal_key


def test_ipfs_put_get_roundtrip():
    s = IPFSimulator()
    cid = s.put(b"hello")
    assert cid.startswith("sha256-")
    assert s.get(cid) == b"hello"


def test_emd_symmetric():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.0, 0.5, 0.5])
    assert abs(emd_histograms(p, q) - emd_histograms(q, p)) < 1e-9


def test_ledger_escrow_and_double_pay_raises():
    ipfs = IPFSimulator()
    L = Ledger(":memory:")
    L.genesis_if_empty({"requester": (10_000, 0), "x": (0, 0)})
    tid, _ = publish_fl_task(
        L,
        ipfs,
        "requester",
        reward=100,
        hosting_fee=20,
        initial_model_bytes=b"m",
        test_dataset_bytes=b"t",
        deadline_block=5,
        task_id="t1",
    )
    row = L.get_task(tid)
    assert row is not None
    assert row.escrow_amount == 120
    assert L.get_account("requester").balance == 10_000 - 120
    L.ensure_account("payout-p1")
    L.complete_task_pay(tid, "pool-a", "payout-p1")
    assert L.get_account("payout-p1").balance == 120
    with pytest.raises(ValueError):
        L.complete_task_pay(tid, "pool-a", "payout-p1")


def test_algorithm3_partition_covers_all():
    rng = np.random.default_rng(0)
    profiles = []
    for i in range(3):
        labels = rng.integers(0, 3, size=50)
        profiles.append(
            TrainerProfile(
                trainer_id=f"t{i}",
                sample_count=50,
                delay=1.0,
                labels=labels,
                num_classes=3,
            )
        )
    pools, _ = algorithm3_pool_formation(
        profiles,
        WelfareParams(train_time=200.0, psi_local_max=20),
        rng=rng,
    )
    assert len(pools) >= 1
    seen = sorted([mid for po in pools for mid in po.member_ids])
    assert seen == sorted([p.trainer_id for p in profiles])


def test_split_hosting_fee():
    assert split_hosting_fee(10, 3) == [4, 3, 3]


def test_test_holdback_roundtrip():
    b = prepare_test_holdback(b"secret-test-data")
    assert reveal_key(b.key, b.key_commitment)
    assert decrypt_test(b.ciphertext, b.key) == b"secret-test-data"


def test_multi_round_gc_survivor():
    com = ["v0", "v1", "v2", "v3"]
    winner = multi_round_graded_consensus(com, ["h1", "h2"], prev_seed="s", rounds=2)
    assert winner is not None
