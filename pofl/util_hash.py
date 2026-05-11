from __future__ import annotations

import hashlib
import json

from pofl.types import TransactionType


def transaction_hash(tx_type: TransactionType, payload_json: str) -> str:
    body = json.dumps({"tx_type": int(tx_type), "payload": json.loads(payload_json)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()
