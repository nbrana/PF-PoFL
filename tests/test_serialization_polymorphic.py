import io

import pytest
import torch

from pofl.fl.client import TinyCNN, TinyMLP, state_dict_to_bytes
from pofl.serialization import (
    decode_model_state_dict_bytes,
    decode_tinymlp_state_dict_bytes,
)


def _bytes(d: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(d, buf)
    return buf.getvalue()


def test_accepts_canonical_mlp():
    m = TinyMLP(8, 4, hidden=16)
    sd = decode_model_state_dict_bytes(state_dict_to_bytes(m.state_dict()))
    assert "fc1.weight" in sd
    assert "conv1.weight" not in sd


def test_accepts_canonical_cnn():
    m = TinyCNN(num_classes=10)
    sd = decode_model_state_dict_bytes(state_dict_to_bytes(m.state_dict()))
    assert "conv1.weight" in sd
    assert "fc.weight" in sd


def test_rejects_mixed_mlp_and_cnn_keys():
    mlp = TinyMLP(8, 4, hidden=16).state_dict()
    cnn = TinyCNN(num_classes=10).state_dict()
    mixed = {**mlp, **cnn}
    with pytest.raises(ValueError, match="ambiguous"):
        decode_model_state_dict_bytes(_bytes(mixed))


def test_rejects_wrong_conv1_shape():
    sd = TinyCNN(num_classes=10).state_dict()
    sd["conv1.weight"] = torch.randn(16, 1, 3, 3)  # wrong out-channels (16 != 32)
    sd["conv1.bias"] = torch.randn(16)
    with pytest.raises(ValueError, match="conv1"):
        decode_model_state_dict_bytes(_bytes(sd))


def test_rejects_wrong_conv2_shape():
    sd = TinyCNN(num_classes=10).state_dict()
    sd["conv2.weight"] = torch.randn(64, 16, 3, 3)  # wrong in-channels
    with pytest.raises(ValueError, match="conv2"):
        decode_model_state_dict_bytes(_bytes(sd))


def test_rejects_wrong_fc_input_size():
    sd = TinyCNN(num_classes=10).state_dict()
    sd["fc.weight"] = torch.randn(10, 800)  # wrong input dim (should be 1600)
    with pytest.raises(ValueError, match="fc.weight"):
        decode_model_state_dict_bytes(_bytes(sd))


def test_decode_tinymlp_rejects_cnn_blob():
    cnn = TinyCNN(num_classes=10)
    blob = state_dict_to_bytes(cnn.state_dict())
    with pytest.raises(ValueError):
        decode_tinymlp_state_dict_bytes(blob)


def test_allow_cnn_false_blocks_cnn():
    cnn = TinyCNN(num_classes=10)
    blob = state_dict_to_bytes(cnn.state_dict())
    with pytest.raises(ValueError, match="not permitted"):
        decode_model_state_dict_bytes(blob, allow_cnn=False)


def test_decode_tinymlp_still_accepts_mlp():
    m = TinyMLP(8, 4, hidden=16)
    sd = decode_tinymlp_state_dict_bytes(state_dict_to_bytes(m.state_dict()))
    assert "fc1.weight" in sd
