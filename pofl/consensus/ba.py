from __future__ import annotations

import hashlib

from pofl.consensus.vrf import committee_ticket_score


def select_committee_weighted_vrf(
    validators: list[tuple[str, int, bytes]],
    seed: str,
    committee_size: int,
) -> list[str]:
    """Select committee: top `committee_size` validators by credit-scaled VRF score."""
    if not validators:
        return []
    k = min(committee_size, len(validators))
    scored: list[tuple[int, str]] = []
    for vid, credit, sk in validators:
        score, _ = committee_ticket_score(sk, vid, seed, credit)
        scored.append((score, vid))
    scored.sort(key=lambda x: -x[0])
    return [v for _, v in scored[:k]]


def ba_lite_finalize(
    committee: list[str],
    candidate_payload_hash: str,
    prev_seed: str,
    honest_fraction: float = 0.85,
) -> bool:
    """Graded voting simulation: each member votes yes with probability honest_fraction.

    Finalize if >= 2/3 vote for the candidate (Algorand-inspired supermajority).
    """
    if not committee:
        return False
    rng_bytes = hashlib.sha256((candidate_payload_hash + prev_seed).encode()).digest()
    threshold_byte = int(honest_fraction * 255)
    votes = 0
    for i, _vid in enumerate(committee):
        b = rng_bytes[i % len(rng_bytes)]
        if b <= threshold_byte:
            votes += 1
    need = (len(committee) * 2 + 2) // 3
    return votes >= need


def multi_round_graded_consensus(
    committee: list[str],
    candidate_hashes: list[str],
    prev_seed: str,
    rounds: int = 3,
    honest_fraction: float = 0.85,
) -> str | None:
    """Multi-round voting: each round Committee narrows to hashes with >= 2/3 votes.

    Returns surviving candidate or None.
    """
    import hashlib

    survivors = list(candidate_hashes)
    if not survivors:
        return None
    for r in range(rounds):
        if len(survivors) == 1:
            return survivors[0]
        scores: dict[str, int] = {h: 0 for h in survivors}
        need = (len(committee) * 2 + 2) // 3
        for h in survivors:
            rng_bytes = hashlib.sha256((h + prev_seed + str(r)).encode()).digest()
            thr = int(honest_fraction * 255)
            votes = 0
            for i, _vid in enumerate(committee):
                b = rng_bytes[i % len(rng_bytes)]
                if b <= thr:
                    votes += 1
            if votes >= need:
                scores[h] = votes
        survivors = [h for h, v in scores.items() if v > 0]
        if not survivors:
            survivors = list(candidate_hashes)
    return survivors[0] if survivors else None
