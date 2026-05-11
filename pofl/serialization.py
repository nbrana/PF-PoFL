from __future__ import annotations

import io
from typing import Any

import torch


def _torch_load_bytes(blob: bytes) -> Any:
    # NOTE: torch.load is pickle-based. We mitigate by strictly validating the
    # resulting object structure and tensor dtypes/shapes. This is still not a
    # full sandbox; prefer migrating to a non-pickle encoding for untrusted data.
    buf = io.BytesIO(blob)
    try:
        return torch.load(buf, map_location="cpu")
    except Exception as e:  # noqa: BLE001 — convert torch/pickle errors to ValueError
        raise ValueError(f"failed to decode tensor blob: {e}") from e


def decode_test_dataset_bytes(
    blob: bytes, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    obj = _torch_load_bytes(blob)
    if not isinstance(obj, dict):
        raise ValueError("test dataset blob must decode to dict")
    if "X" not in obj:
        raise ValueError("missing key: X")
    if "y" not in obj:
        raise ValueError("missing key: y")

    X = obj["X"]
    y = obj["y"]
    if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("X and y must be torch.Tensor")
    if X.dim() not in (2, 4):
        raise ValueError("X must be 2D (N, D) or 4D (N, C, H, W)")
    if y.dim() != 1:
        raise ValueError("y must be 1D (N,)")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same length")
    if X.dtype not in (torch.float32, torch.float64):
        raise ValueError("X must be float tensor")
    if y.dtype != torch.long:
        raise ValueError("y must be int64 (torch.long)")

    return X.to(device), y.to(device)


_MLP_REQUIRED = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
_CNN_REQUIRED = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc.weight",
    "fc.bias",
)


def _validate_mlp_state_dict(sd: dict[str, torch.Tensor]) -> None:
    for k in _MLP_REQUIRED:
        if k not in sd:
            raise ValueError(f"missing key: {k}")
    w1 = sd["fc1.weight"]
    b1 = sd["fc1.bias"]
    w2 = sd["fc2.weight"]
    b2 = sd["fc2.bias"]
    if w1.dim() != 2 or w2.dim() != 2:
        raise ValueError("fc weights must be 2D")
    if b1.dim() != 1 or b2.dim() != 1:
        raise ValueError("fc biases must be 1D")
    if w1.shape[0] != b1.shape[0]:
        raise ValueError("fc1 bias mismatch")
    if w2.shape[0] != b2.shape[0]:
        raise ValueError("fc2 bias mismatch")
    if w2.shape[1] != w1.shape[0]:
        raise ValueError("hidden dim mismatch between fc1 and fc2")
    for k in _MLP_REQUIRED:
        if sd[k].dtype not in (torch.float32, torch.float64):
            raise ValueError("weights must be float tensors")


def _validate_cnn_state_dict(sd: dict[str, torch.Tensor]) -> None:
    for k in _CNN_REQUIRED:
        if k not in sd:
            raise ValueError(f"missing key: {k}")
    w_c1 = sd["conv1.weight"]
    b_c1 = sd["conv1.bias"]
    w_c2 = sd["conv2.weight"]
    b_c2 = sd["conv2.bias"]
    w_fc = sd["fc.weight"]
    b_fc = sd["fc.bias"]
    if tuple(w_c1.shape) != (32, 1, 3, 3):
        raise ValueError("conv1.weight must be (32,1,3,3)")
    if tuple(b_c1.shape) != (32,):
        raise ValueError("conv1.bias must be (32,)")
    if tuple(w_c2.shape) != (64, 32, 3, 3):
        raise ValueError("conv2.weight must be (64,32,3,3)")
    if tuple(b_c2.shape) != (64,):
        raise ValueError("conv2.bias must be (64,)")
    if w_fc.dim() != 2 or w_fc.shape[1] != 64 * 5 * 5:
        raise ValueError("fc.weight must be (num_classes, 1600)")
    if b_fc.dim() != 1 or b_fc.shape[0] != w_fc.shape[0]:
        raise ValueError("fc.bias must align with fc.weight")
    for k in _CNN_REQUIRED:
        if sd[k].dtype not in (torch.float32, torch.float64):
            raise ValueError("weights must be float tensors")


def decode_model_state_dict_bytes(
    blob: bytes, *, allow_cnn: bool = True
) -> dict[str, torch.Tensor]:
    """Polymorphic decoder: accepts MLP or CNN state dicts.

    Mixed key sets are rejected so callers cannot smuggle extra layers past
    the validator. Shape gates pin the CNN topology to TinyCNN's exact layout.
    """
    obj = _torch_load_bytes(blob)
    if not isinstance(obj, dict):
        raise ValueError("weights blob must decode to dict[str, Tensor]")
    sd: dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, torch.Tensor):
            sd[k] = v

    has_cnn_keys = any(k in sd for k in _CNN_REQUIRED)
    has_mlp_keys = any(k in sd for k in _MLP_REQUIRED)
    if has_cnn_keys and has_mlp_keys:
        raise ValueError("ambiguous state dict: mixes MLP and CNN keys")
    if has_cnn_keys:
        if not allow_cnn:
            raise ValueError("CNN state dict not permitted by caller")
        _validate_cnn_state_dict(sd)
        return sd
    _validate_mlp_state_dict(sd)
    return sd


def decode_tinymlp_state_dict_bytes(blob: bytes) -> dict[str, torch.Tensor]:
    """Back-compat wrapper for callers that only accept MLP state dicts."""
    sd = decode_model_state_dict_bytes(blob, allow_cnn=False)
    return sd
