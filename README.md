# PF-PoFL (Platform-Free Proof of Federated Learning)

Proof of Concept implementation of PF-PoFL paper
Original Paper is Y. Wang, H. Peng, Z. Su, T. H. Luan, A. Benslimane and Y. Wu, "A Platform-Free Proof of Federated Learning Consensus Mechanism for Sustainable Blockchains," in IEEE Journal on Selected Areas in Communications, vol. 40, no. 12, pp. 3305-3324, Dec. 2022, doi: 10.1109/JSAC.2022.3213347.

## Quickstart (Python demo)

```bash
uv sync --extra dev
uv run python scripts/run_simulation.py --task synthetic
```

MNIST demo

```bash
uv run python scripts/run_simulation.py \
  --task mnist \
  --num-trainers 4 \
  --samples-per-trainer 200 \
  --test-samples 1000 \
  --rounds 3 \
  --local-epochs 1 \
  --hidden 128 \
  --sigma 0.0
```


### MNIST data (offline-friendly)

By default, MNIST is cached under `.data/` (or override with `--mnist-root`). If downloads are blocked, place these files in that directory:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`


## Substrate / FRAME

See [substrate/README.md](substrate/README.md) for the solochain template, `pallet-pf-pofl`, and `cargo check` targets.

## Docker (full node, RocksDB)

```bash
docker build -t pf-pofl-node .
docker run --rm pf-pofl-node -- --help
```

Requires Docker; first build compiles the full Substrate node (15–40+ minutes typical). See `[Dockerfile](Dockerfile)`.

```bash
uv run python scripts/eval_suite.py
```

## Run the demo

```bash
uv sync --extra dev
uv run python scripts/run_simulation.py 
```

## Run tests

```bash
uv run pytest tests/ -q
```
