from __future__ import annotations

import hashlib
import json
import uuid

from pofl.ledger import Ledger
from pofl.types import FLModelPayload, TaskPublicationPayload, TransactionType
from pofl.util_hash import transaction_hash


def publish_fl_task(
    ledger: Ledger,
    ipfs,
    publisher: str,
    reward: int,
    hosting_fee: int,
    initial_model_bytes: bytes,
    test_dataset_bytes: bytes,
    deadline_block: int,
    task_id: str | None = None,
    *,
    participation_deposit: int = 0,
    release_block: int = 0,
    delta_test_blocks: int = 0,
) -> tuple[str, str]:
    """Publish task: commits SHA-256 of test dataset, escrows reward+hosting_fee.

    Returns (task_id, tx_hash).
    """
    test_hash = hashlib.sha256(test_dataset_bytes).hexdigest()
    model_cid = ipfs.put(initial_model_bytes)
    test_cid = ipfs.put(test_dataset_bytes)
    tid = task_id or f"task-{uuid.uuid4().hex[:12]}"
    created_block = max(0, ledger.tip_height())
    payload = TaskPublicationPayload(
        publisher=publisher,
        reward=reward,
        hosting_fee=hosting_fee,
        initial_model_cid=model_cid,
        test_dataset_cid=test_cid,
        test_dataset_hash=test_hash,
        deadline_block=deadline_block,
        task_id=tid,
        participation_deposit=participation_deposit,
        release_block=release_block,
        delta_test_blocks=delta_test_blocks,
    )
    pjson = payload.to_json()
    ledger.publish_task(
        task_id=tid,
        publisher=publisher,
        reward=reward,
        hosting_fee=hosting_fee,
        initial_model_cid=model_cid,
        test_dataset_cid=test_cid,
        test_dataset_hash=test_hash,
        deadline_block=deadline_block,
        created_block=created_block,
        participation_deposit=participation_deposit,
        release_block=release_block,
        delta_test_blocks=delta_test_blocks,
    )
    txh = transaction_hash(TransactionType.TASK_PUBLICATION, pjson)
    ledger.insert_transaction(txh, TransactionType.TASK_PUBLICATION, pjson)
    return tid, txh


def submit_payment_tx(
    ledger: Ledger,
    from_account: str,
    to_account: str,
    amount: int,
    fee: int = 0,
) -> str:
    from pofl.types import PaymentPayload

    ledger.apply_payment(from_account, to_account, amount, fee)
    payload = PaymentPayload(
        from_account=from_account, to_account=to_account, amount=amount, fee=fee
    )
    pjson = payload.to_json()
    txh = transaction_hash(TransactionType.PAYMENT, pjson)
    ledger.insert_transaction(txh, TransactionType.PAYMENT, pjson)
    return txh


def submit_fl_model_tx(
    ledger: Ledger,
    pool_id: str,
    task_id: str,
    round_index: int,
    weights_cid: str,
    member_ids: list[str],
    *,
    submission_block: int = 0,
    participation_deposit: int = 0,
    reference_score: float | None = None,
) -> str:
    payload = FLModelPayload(
        submitter_pool_id=pool_id,
        task_id=task_id,
        round_index=round_index,
        weights_cid=weights_cid,
        member_ids_json=json.dumps(sorted(member_ids)),
        submission_block=submission_block,
        participation_deposit=participation_deposit,
        reference_score=reference_score,
    )
    pjson = payload.to_json()
    txh = transaction_hash(TransactionType.FL_MODEL, pjson)
    ledger.insert_transaction(txh, TransactionType.FL_MODEL, pjson)
    ledger.record_fl_submission(
        task_id,
        pool_id,
        round_index,
        weights_cid,
        txh,
        submission_block=submission_block,
        participation_deposit=participation_deposit,
    )
    return txh
