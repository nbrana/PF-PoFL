import io

import pytest
import torch


def _dataset_bytes(X: torch.Tensor, y: torch.Tensor) -> bytes:
    b = io.BytesIO()
    torch.save({"X": X, "y": y}, b)
    return b.getvalue()


def test_decode_test_dataset_rejects_missing_keys() -> None:
    from pofl.serialization import decode_test_dataset_bytes

    b = io.BytesIO()
    torch.save({"Z": torch.zeros(2, 3), "y": torch.zeros(2, dtype=torch.long)}, b)

    with pytest.raises(ValueError, match="missing key"):
        decode_test_dataset_bytes(b.getvalue(), device=torch.device("cpu"))


def test_decode_test_dataset_rejects_shape_mismatch() -> None:
    from pofl.serialization import decode_test_dataset_bytes

    X = torch.zeros(10, 784, dtype=torch.float32)
    y = torch.zeros(9, dtype=torch.long)
    blob = _dataset_bytes(X, y)

    with pytest.raises(ValueError, match="same length"):
        decode_test_dataset_bytes(blob, device=torch.device("cpu"))


def test_decode_state_dict_rejects_missing_fc_keys() -> None:
    from pofl.serialization import decode_tinymlp_state_dict_bytes

    bad = {"something": torch.zeros(1)}
    b = io.BytesIO()
    torch.save(bad, b)

    with pytest.raises(ValueError, match="missing key"):
        decode_tinymlp_state_dict_bytes(b.getvalue())

