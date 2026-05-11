"""Deterministic off-chain scoring helpers (mirrors validator ranking, no IPFS I/O).

Used as reference logic for Substrate OCW or external workers.
"""

from __future__ import annotations

import io

import torch

from pofl.fl.client import model_from_state_dict
from pofl.fl.server import evaluate_loss


def score_model_bytes(state_dict_bytes: bytes, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    buf = io.BytesIO(state_dict_bytes)
    sd = torch.load(buf, map_location="cpu")
    model = model_from_state_dict(sd)
    return evaluate_loss(model, X, y, device)


def ranking_from_submissions(
    cid_to_bytes: dict[str, bytes],
    pool_to_cid: dict[str, str],
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for pool_id, cid in pool_to_cid.items():
        loss = score_model_bytes(cid_to_bytes[cid], X, y, device)
        ranked.append((pool_id, loss))
    ranked.sort(key=lambda x: x[1])
    return ranked
