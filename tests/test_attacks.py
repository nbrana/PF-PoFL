import io
import os

import pytest
import torch

from pofl.fl.client import TinyMLP, state_dict_to_bytes, train_local_sgd
from pofl.fl.server import evaluate_loss
from pofl.ipfs_sim import IPFSimulator
from pofl.ledger import Ledger
from pofl.roles.requester import publish_fl_task
from pofl.roles.validator import rank_models_for_task
from pofl.test_holdback import decrypt_test, prepare_test_holdback


def _save_dataset_bytes(X: torch.Tensor, y: torch.Tensor) -> bytes:
    b = io.BytesIO()
    torch.save({"X": X, "y": y}, b)
    return b.getvalue()


def _vprint(msg: str) -> None:
    if os.getenv("PF_POFL_TEST_VERBOSE") in ("1", "true", "TRUE", "yes", "YES"):
        print(msg)


def test_artifact_tampering_rejected_by_test_hash_commitment() -> None:
    """If the stored test blob is modified after publication, validators must reject it."""
    torch.manual_seed(0)

    ledger = Ledger(":memory:")
    ipfs = IPFSimulator()
    ledger.genesis_if_empty({"requester": (10_000, 0)})

    model = TinyMLP(8, 4, hidden=16)
    init_bytes = state_dict_to_bytes(model.state_dict())

    X_test = torch.randn(32, 8)
    y_test = torch.randint(0, 4, (32,), dtype=torch.long)
    test_bytes = _save_dataset_bytes(X_test, y_test)

    task_id, _tx = publish_fl_task(
        ledger,
        ipfs,
        publisher="requester",
        reward=100,
        hosting_fee=10,
        initial_model_bytes=init_bytes,
        test_dataset_bytes=test_bytes,
        deadline_block=5,
        task_id="t-tamper",
    )
    task = ledger.get_task(task_id)
    assert task is not None
    _vprint(f"task_id={task_id} test_cid={task.test_dataset_cid} committed_hash={task.test_dataset_hash}")

    # Attacker tampers with the blob stored under the committed CID.
    tampered = _save_dataset_bytes(X_test + 999.0, y_test)
    ipfs._mem[task.test_dataset_cid] = tampered  # type: ignore[attr-defined]
    _vprint(f"tampered_bytes_len={len(tampered)} (replaced blob under committed CID)")

    with pytest.raises(ValueError, match="hash mismatch"):
        rank_models_for_task(
            ledger,
            ipfs,
            task_id,
            device=torch.device("cpu"),
            current_block=task.deadline_block,
            ranking_window_delta=4,
        )


def test_training_spoofing_overfit_when_test_is_available() -> None:
    """Demonstrate the threat: if a trainer can access the test set, they can overfit it."""
    torch.manual_seed(0)
    device = torch.device("cpu")

    input_dim = 16
    num_classes = 3
    n_train = 96
    n_test = 64

    X_train = torch.randn(n_train, input_dim)
    y_train = torch.randint(0, num_classes, (n_train,), dtype=torch.long)
    X_test = torch.randn(n_test, input_dim)
    y_test = torch.randint(0, num_classes, (n_test,), dtype=torch.long)

    # Honest trainer trains on train set.
    honest = TinyMLP(input_dim, num_classes, hidden=32)
    train_local_sgd(honest, X_train, y_train, epochs=5, batch_size=32, lr=0.2, device=device)
    honest_loss = evaluate_loss(honest, X_test, y_test, device)

    # Spoofing trainer trains directly on the test set they should not have.
    spoofer = TinyMLP(input_dim, num_classes, hidden=32)
    train_local_sgd(spoofer, X_test, y_test, epochs=20, batch_size=32, lr=0.2, device=device)
    spoofer_loss = evaluate_loss(spoofer, X_test, y_test, device)

    _vprint(f"honest_loss_on_test={honest_loss:.6f}")
    _vprint(f"spoofer_loss_on_test={spoofer_loss:.6f}")
    _vprint(f"loss_improvement={(honest_loss - spoofer_loss):.6f}")
    assert spoofer_loss < honest_loss


def test_holdback_ciphertext_does_not_reveal_test_plaintext() -> None:
    """Holdback bundle is reversible only with the key; ciphertext alone should not match plaintext."""
    torch.manual_seed(0)

    X = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,), dtype=torch.long)
    plaintext = _save_dataset_bytes(X, y)

    b = prepare_test_holdback(plaintext)
    _vprint(f"holdback_key_commitment={b.key_commitment} ciphertext_len={len(b.ciphertext)}")
    assert b.ciphertext != plaintext
    assert decrypt_test(b.ciphertext, b.key) == plaintext

