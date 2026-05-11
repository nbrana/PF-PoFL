from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MnistTrainerPartition:
    trainer_id: str
    X: torch.Tensor
    y: torch.Tensor
    labels: np.ndarray


def flatten_mnist_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or tuple(x.shape[1:]) != (1, 28, 28):
        raise ValueError("expected (N,1,28,28)")
    return x.reshape(x.shape[0], 28 * 28).to(dtype=torch.float32)


def _read_idx_gz(path: str) -> torch.Tensor:
    import gzip
    import struct

    with gzip.open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError("IDX file too small")

    magic = struct.unpack(">I", data[:4])[0]
    if magic == 2049:  # labels
        n = struct.unpack(">I", data[4:8])[0]
        buf = memoryview(data)[8:]
        if len(buf) != n:
            raise ValueError("IDX label length mismatch")
        # Use a writable buffer to avoid PyTorch warnings.
        return torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
    if magic == 2051:  # images
        if len(data) < 16:
            raise ValueError("IDX image header too small")
        n, rows, cols = struct.unpack(">III", data[4:16])
        buf = memoryview(data)[16:]
        exp = n * rows * cols
        if len(buf) != exp:
            raise ValueError("IDX image length mismatch")
        t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).clone()
        return t.view(n, rows, cols)

    raise ValueError("unsupported IDX magic")


def load_mnist_tensors(
    *,
    data_root: str,
    train_limit: int | None = None,
    test_limit: int | None = None,
    train_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    test_override: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if train_override is not None:
        xtr, ytr = train_override
    else:
        try:
            from torchvision import datasets, transforms

            ds = datasets.MNIST(
                root=data_root, train=True, download=True, transform=transforms.ToTensor()
            )
            xtr = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
            ytr = torch.tensor([int(ds[i][1]) for i in range(len(ds))], dtype=torch.long)
        except Exception:
            # Network may be unavailable in some environments; support local IDX files.
            img_path = f"{data_root}/train-images-idx3-ubyte.gz"
            lbl_path = f"{data_root}/train-labels-idx1-ubyte.gz"
            try:
                imgs = _read_idx_gz(img_path).to(dtype=torch.float32) / 255.0
                labs = _read_idx_gz(lbl_path).to(dtype=torch.long)
            except FileNotFoundError as e:
                raise RuntimeError(
                    "MNIST download failed and local IDX files were not found. "
                    f"Place MNIST gzip IDX files in '{data_root}': "
                    "train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, "
                    "t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz."
                ) from e
            xtr = imgs.unsqueeze(1)
            ytr = labs

    if test_override is not None:
        xte, yte = test_override
    else:
        try:
            from torchvision import datasets, transforms

            ds = datasets.MNIST(
                root=data_root, train=False, download=True, transform=transforms.ToTensor()
            )
            xte = torch.stack([ds[i][0] for i in range(len(ds))], dim=0)
            yte = torch.tensor([int(ds[i][1]) for i in range(len(ds))], dtype=torch.long)
        except Exception:
            img_path = f"{data_root}/t10k-images-idx3-ubyte.gz"
            lbl_path = f"{data_root}/t10k-labels-idx1-ubyte.gz"
            try:
                imgs = _read_idx_gz(img_path).to(dtype=torch.float32) / 255.0
                labs = _read_idx_gz(lbl_path).to(dtype=torch.long)
            except FileNotFoundError as e:
                raise RuntimeError(
                    "MNIST download failed and local IDX files were not found. "
                    f"Place MNIST gzip IDX files in '{data_root}': "
                    "train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, "
                    "t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz."
                ) from e
            xte = imgs.unsqueeze(1)
            yte = labs

    if train_limit is not None:
        xtr = xtr[:train_limit]
        ytr = ytr[:train_limit]
    if test_limit is not None:
        xte = xte[:test_limit]
        yte = yte[:test_limit]

    Xtr = flatten_mnist_batch(xtr)
    Xte = flatten_mnist_batch(xte)
    return Xtr, ytr.to(dtype=torch.long), Xte, yte.to(dtype=torch.long)


def load_mnist_tensors_4d(
    *,
    data_root: str,
    train_limit: int | None = None,
    test_limit: int | None = None,
    train_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    test_override: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as load_mnist_tensors but returns (N,1,28,28) for CNN inputs."""
    Xtr_flat, ytr, Xte_flat, yte = load_mnist_tensors(
        data_root=data_root,
        train_limit=train_limit,
        test_limit=test_limit,
        train_override=train_override,
        test_override=test_override,
    )
    Xtr = Xtr_flat.reshape(-1, 1, 28, 28).to(dtype=torch.float32)
    Xte = Xte_flat.reshape(-1, 1, 28, 28).to(dtype=torch.float32)
    return Xtr, ytr, Xte, yte


def partition_mnist_tensors(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    num_trainers: int,
    samples_per_trainer: int,
    num_classes: int,
    seed: int,
) -> list[MnistTrainerPartition]:
    rng = np.random.default_rng(seed)
    n = int(X.shape[0])
    if y.shape[0] != n:
        raise ValueError("X/y size mismatch")
    if X.dim() not in (2, 4):
        raise ValueError("X must be 2D (N, D) or 4D (N, 1, 28, 28)")
    if y.dim() != 1:
        raise ValueError("y must be 1D (N,)")
    if X.dim() == 2 and int(X.shape[1]) != 28 * 28:
        raise ValueError("X must have 784 features for flattened MNIST")
    if X.dim() == 4 and tuple(X.shape[1:]) != (1, 28, 28):
        raise ValueError("X must be (N,1,28,28) for CNN MNIST")
    if num_classes != 10:
        raise ValueError("MNIST expects num_classes=10")

    idx = rng.permutation(n)
    need = num_trainers * samples_per_trainer
    if need > n:
        raise ValueError("not enough samples for requested partition")
    idx = idx[:need]

    Xs = X[idx]
    ys = y[idx]

    out: list[MnistTrainerPartition] = []
    for i in range(num_trainers):
        sl = slice(i * samples_per_trainer, (i + 1) * samples_per_trainer)
        Xi = Xs[sl].clone()
        yi = ys[sl].clone()
        labels = yi.detach().cpu().numpy().astype(np.int64, copy=True)
        out.append(MnistTrainerPartition(trainer_id=f"trainer-{i}", X=Xi, y=yi, labels=labels))
    return out

