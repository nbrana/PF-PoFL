import torch

from pofl.data.mnist import _read_idx_gz, flatten_mnist_batch, load_mnist_tensors, partition_mnist_tensors


def test_partition_shapes_and_dtypes() -> None:
    X = torch.randn(100, 784, dtype=torch.float32)
    y = torch.randint(0, 10, (100,), dtype=torch.long)

    parts = partition_mnist_tensors(
        X, y, num_trainers=4, samples_per_trainer=20, num_classes=10, seed=0
    )

    assert len(parts) == 4
    for p in parts:
        assert p.X.dtype == torch.float32
        assert p.y.dtype == torch.long
        assert p.X.shape == (20, 784)
        assert p.y.shape == (20,)
        assert p.labels.shape == (20,)
        assert (torch.tensor(p.labels, dtype=torch.long) == p.y.cpu()).all()


def test_partition_is_reproducible_with_seed() -> None:
    X = torch.randn(120, 784, dtype=torch.float32)
    y = torch.randint(0, 10, (120,), dtype=torch.long)

    a = partition_mnist_tensors(
        X, y, num_trainers=3, samples_per_trainer=30, num_classes=10, seed=123
    )
    b = partition_mnist_tensors(
        X, y, num_trainers=3, samples_per_trainer=30, num_classes=10, seed=123
    )

    for pa, pb in zip(a, b, strict=True):
        assert torch.allclose(pa.X, pb.X)
        assert torch.equal(pa.y, pb.y)
        assert (pa.labels == pb.labels).all()


def test_flatten_mnist_batch() -> None:
    X = torch.rand(5, 1, 28, 28, dtype=torch.float32)
    out = flatten_mnist_batch(X)
    assert out.shape == (5, 784)
    assert out.dtype == torch.float32


def test_load_mnist_tensors_shapes_with_fake_dataset() -> None:
    # Avoid network/download in unit tests by passing fake tensors.
    fake_train = (torch.rand(7, 1, 28, 28), torch.randint(0, 10, (7,), dtype=torch.long))
    fake_test = (torch.rand(3, 1, 28, 28), torch.randint(0, 10, (3,), dtype=torch.long))

    Xtr, ytr, Xte, yte = load_mnist_tensors(
        data_root="/tmp/does-not-matter",
        train_limit=None,
        test_limit=None,
        train_override=fake_train,
        test_override=fake_test,
    )

    assert Xtr.shape == (7, 784)
    assert ytr.shape == (7,)
    assert Xte.shape == (3, 784)
    assert yte.shape == (3,)


def test_read_idx_gz_parses_images_and_labels(tmp_path) -> None:
    import gzip
    import struct

    # Minimal IDX image file: 2 images of 2x2.
    images = bytes([0, 1, 2, 3, 4, 5, 6, 7])
    img_hdr = struct.pack(">IIII", 2051, 2, 2, 2)
    img_path = tmp_path / "train-images-idx3-ubyte.gz"
    with gzip.open(img_path, "wb") as f:
        f.write(img_hdr + images)

    # Minimal IDX label file: 2 labels.
    lbl_hdr = struct.pack(">II", 2049, 2)
    lbl_path = tmp_path / "train-labels-idx1-ubyte.gz"
    with gzip.open(lbl_path, "wb") as f:
        f.write(lbl_hdr + bytes([7, 9]))

    X = _read_idx_gz(str(img_path))
    y = _read_idx_gz(str(lbl_path))

    assert X.shape == (2, 2, 2)
    assert X.dtype == torch.uint8
    assert y.shape == (2,)
    assert y.dtype == torch.uint8

