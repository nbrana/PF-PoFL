from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any


class TransactionType(IntEnum):
    PAYMENT = 0
    TASK_PUBLICATION = 1
    FL_MODEL = 2


@dataclass
class PaymentPayload:
    from_account: str
    to_account: str
    amount: int
    fee: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> PaymentPayload:
        d = json.loads(s)
        return PaymentPayload(**d)


@dataclass
class TaskPublicationPayload:
    publisher: str
    reward: int
    hosting_fee: int
    initial_model_cid: str
    test_dataset_cid: str
    test_dataset_hash: str
    deadline_block: int
    task_id: str
    participation_deposit: int = 0
    release_block: int = 0
    delta_test_blocks: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> TaskPublicationPayload:
        d = json.loads(s)
        return TaskPublicationPayload(**d)


@dataclass
class FLModelPayload:
    submitter_pool_id: str
    task_id: str
    round_index: int
    weights_cid: str
    member_ids_json: str
    participation_deposit: int = 0
    submission_block: int = 0
    reference_score: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> FLModelPayload:
        d = json.loads(s)
        return FLModelPayload(**d)


@dataclass
class Transaction:
    tx_type: TransactionType
    payload: dict[str, Any]

    def payload_json(self) -> str:
        return json.dumps(self.payload, sort_keys=True)


@dataclass
class Block:
    height: int
    prev_hash: str
    merkle_root: str
    proposer: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "height": self.height,
                "prev_hash": self.prev_hash,
                "merkle_root": self.merkle_root,
                "proposer": self.proposer,
                "payload": self.payload,
            },
            sort_keys=True,
        )
