import torch

from pofl.fl.client import TinyCNN, model_from_state_dict, state_dict_to_bytes
from pofl.fl.server import evaluate_loss
from pofl.serialization import decode_model_state_dict_bytes


def test_cnn_forward_shape():
    m = TinyCNN(num_classes=10)
    x = torch.randn(7, 1, 28, 28)
    out = m(x)
    assert out.shape == (7, 10)


def test_cnn_state_dict_keys():
    m = TinyCNN(num_classes=10)
    sd = m.state_dict()
    for k in (
        "conv1.weight",
        "conv1.bias",
        "conv2.weight",
        "conv2.bias",
        "fc.weight",
        "fc.bias",
    ):
        assert k in sd


def test_cnn_roundtrip_identical_logits():
    torch.manual_seed(0)
    m = TinyCNN(num_classes=10)
    blob = state_dict_to_bytes(m.state_dict())
    sd = decode_model_state_dict_bytes(blob)
    m2 = model_from_state_dict(sd)

    x = torch.randn(3, 1, 28, 28)
    m.eval()
    m2.eval()
    with torch.no_grad():
        a = m(x)
        b = m2(x)
    assert torch.allclose(a, b)


def test_model_from_state_dict_dispatches_on_keys():
    cnn = TinyCNN(num_classes=10)
    cnn_blob = state_dict_to_bytes(cnn.state_dict())
    cnn_sd = decode_model_state_dict_bytes(cnn_blob)
    assert isinstance(model_from_state_dict(cnn_sd), TinyCNN)


def test_evaluate_loss_works_for_cnn():
    torch.manual_seed(0)
    m = TinyCNN(num_classes=10)
    X = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,), dtype=torch.long)
    loss = evaluate_loss(m, X, y, torch.device("cpu"))
    assert loss > 0
