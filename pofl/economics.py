"""Participation rewards/slashing helpers (paper Sec. III-C; Python PoC)."""

from __future__ import annotations

import math
import statistics
from typing import Sequence


def redeem_deposit(loss: float, threshold: float) -> bool:
    """Return True if participant should redeem deposit (met performance bar)."""
    return loss <= threshold


def slash_fraction(loss: float, threshold: float, max_slash: float = 0.5) -> float:
    """Linear stub: excess loss above threshold maps to slash fraction capped at max_slash."""
    if loss <= threshold:
        return 0.0
    excess = loss - threshold
    return min(max_slash, excess / max(threshold, 1e-6))


def split_hosting_fee(hosting_fee: int, num_validators: int) -> list[int]:
    """Equal split; remainder assigned to earlier indices."""
    if num_validators <= 0:
        return []
    base = hosting_fee // num_validators
    rem = hosting_fee % num_validators
    return [base + (1 if i < rem else 0) for i in range(num_validators)]


def quartile_threshold(losses: Sequence[float]) -> float:
    """Paper "preset threshold (e.g. first quartile)" used by Alg. 2 line 5.

    Returns the 25th percentile of the loss list (lower = better). Pools with
    loss <= this threshold are in W_τ. Empty input -> +inf (no qualifying).
    """
    arr = [float(x) for x in losses if x is not None and not math.isnan(x)]
    if not arr:
        return math.inf
    if len(arr) == 1:
        return arr[0]
    arr_sorted = sorted(arr)
    # Type 7 quantile (numpy/R default) — linear interpolation.
    idx = 0.25 * (len(arr_sorted) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return arr_sorted[lo]
    frac = idx - lo
    return arr_sorted[lo] + (arr_sorted[hi] - arr_sorted[lo]) * frac


def validator_share(
    xi1: int, xi2: int, w_minus_winner_size: int, num_validators: int
) -> tuple[int, int]:
    """Paper Alg. 2 line 4 share per validator: (xi1 + xi2*|W̄|/|V|).

    Returns (per_validator_amount, remainder) where remainder is distributed
    to the first `remainder` validators by `split_hosting_fee` semantics.
    """
    if num_validators <= 0:
        return 0, 0
    if xi1 < 0 or xi2 < 0 or w_minus_winner_size < 0:
        raise ValueError("negative input to validator_share")
    redistributed_xi2 = (xi2 * w_minus_winner_size) // max(num_validators, 1)
    total_per = redistributed_xi2 + (xi1 // num_validators)
    rem = xi1 % num_validators
    return int(total_per), int(rem)


def median_norm(norms: Sequence[float]) -> float:
    """Paper Theorem 1 sensitivity bound A = median{||ΔΘ_m||₂ : m ∈ M_j ∪ {φ_j}}."""
    if not norms:
        return 0.0
    return float(statistics.median(norms))
