from __future__ import annotations

import io

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TinyCNN(nn.Module):
    """3-layer CNN for 28x28 grayscale inputs (paper Table I MNIST architecture).

    Layer widths chosen for CPU runnability: 32 and 64 conv channels with 3x3
    kernels and 2x2 max-pool. Three weight-bearing layers total: conv1, conv2,
    fc. Forward expects (N, 1, 28, 28); output is (N, num_classes).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc(x)


def state_dict_to_bytes(state: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def state_dict_from_bytes(data: bytes) -> dict[str, torch.Tensor]:
    # Legacy: used for internal, trusted roundtrips. For untrusted submissions,
    # use `pofl.serialization.decode_model_state_dict_bytes`.
    buf = io.BytesIO(data)
    obj = torch.load(buf, map_location="cpu")
    return obj  # type: ignore[return-value]


def model_from_state_dict(sd: dict[str, torch.Tensor]) -> nn.Module:
    """Dispatch by state-dict keys: TinyCNN (conv1.weight present) else TinyMLP."""
    if "conv1.weight" in sd:
        w_fc = sd["fc.weight"]
        num_classes = int(w_fc.shape[0])
        m_cnn: nn.Module = TinyCNN(num_classes=num_classes)
        m_cnn.load_state_dict(sd)
        return m_cnn
    w1 = sd["fc1.weight"]
    w2 = sd["fc2.weight"]
    input_dim = int(w1.shape[1])
    hidden = int(w1.shape[0])
    num_classes = int(w2.shape[0])
    m_mlp = TinyMLP(input_dim, num_classes, hidden)
    m_mlp.load_state_dict(sd)
    return m_mlp


def train_local_sgd(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    model.train()
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = X.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X[idx]
            yb = y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
