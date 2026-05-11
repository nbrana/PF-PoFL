from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class VRFOutput:
    value: int
    proof_hex: str


def vrf_evaluate(secret: bytes, role: str, seed: str) -> VRFOutput:
    """PoC VRF: deterministic output from (secret, role, seed). Not a real VRF."""
    h = hashlib.sha256(secret + role.encode("utf-8") + seed.encode("utf-8")).hexdigest()
    value = int(h[:12], 16)
    return VRFOutput(value=value, proof_hex=h)


def committee_ticket_score(
    secret: bytes,
    validator_id: str,
    seed: str,
    credit: int,
) -> tuple[int, VRFOutput]:
    out = vrf_evaluate(secret, f"committee:{validator_id}", seed)
    score = out.value * max(1, 1 + credit)
    return score, out
