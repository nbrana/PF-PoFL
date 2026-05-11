from __future__ import annotations

import copy

import torch
import torch.nn as nn


def flatten_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_to_model(vec: torch.Tensor, template: nn.Module) -> None:
    offset = 0
    for p in template.parameters():
        n = p.numel()
        p.data.copy_(vec[offset : offset + n].view_as(p.data))
        offset += n


def user_contribution_vector(
    global_vec: torch.Tensor,
    local_model: nn.Module,
) -> torch.Tensor:
    local_vec = flatten_grads(local_model)
    return local_vec - global_vec


def clip_tensor_l2(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = vec.norm().item()
    if n <= max_norm or n == 0:
        return vec
    return vec * (max_norm / n)


def aggregate_user_updates_weighted(
    global_model: nn.Module,
    local_models: list[nn.Module],
    user_weights: list[float],
    clip_norm: float,
) -> nn.Module:
    """Average clipped (local - global) deltas, apply to a copy of global."""
    global_vec = flatten_grads(global_model)
    deltas: list[torch.Tensor] = []
    for lm, w in zip(local_models, user_weights, strict=True):
        delta = user_contribution_vector(global_vec, lm)
        deltas.append(clip_tensor_l2(delta, clip_norm) * w)
    stacked = torch.stack(deltas, dim=0)
    tot_w = sum(user_weights)
    avg_delta = stacked.sum(dim=0) / max(tot_w, 1e-8)
    new_global = copy.deepcopy(global_model)
    new_vec = global_vec + avg_delta
    unflatten_to_model(new_vec, new_global)
    return new_global


def add_gaussian_noise_to_model(model: nn.Module, sigma: float, sensitivity: float) -> None:
    scale = sigma * sensitivity
    with torch.no_grad():
        for p in model.parameters():
            noise = torch.randn_like(p.data) * scale
            p.data.add_(noise)


def evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
    return float(acc)


def evaluate_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    import torch.nn.functional as F

    model.eval()
    model.to(device)
    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y).item()
    return float(loss)
