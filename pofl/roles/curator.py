from __future__ import annotations

import copy

import torch

from pofl.fl.client import TinyMLP, train_local_sgd
from pofl.fl.pool_formation import MiningPool
from pofl.fl.server import add_gaussian_noise_to_model, aggregate_user_updates_weighted


def run_pool_federated_rounds(
    pool: MiningPool,
    global_model: TinyMLP,
    data_by_trainer: dict[str, tuple[torch.Tensor, torch.Tensor]],
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    sigma: float,
    sensitivity: float,
    device: torch.device,
) -> TinyMLP:
    """Curator-coordinated FL with user-level clipping and global Gaussian noise."""
    g = global_model
    for _ in range(rounds):
        locals_: list[TinyMLP] = []
        weights: list[float] = []
        for pid in pool.member_ids:
            X, y = data_by_trainer[pid]
            X = X.to(device)
            y = y.to(device)
            lm = copy.deepcopy(g)
            train_local_sgd(lm, X, y, local_epochs, batch_size, lr, device)
            locals_.append(lm)
            prof = next(p for p in pool.profiles if p.trainer_id == pid)
            weights.append(float(prof.sample_count))
        g = aggregate_user_updates_weighted(g, locals_, weights, clip_norm)
        add_gaussian_noise_to_model(g, sigma, sensitivity)
    return g
