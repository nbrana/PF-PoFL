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

## Python ↔ Substrate bridge

`pofl/chain/` wraps every `pallet-pf-pofl` dispatchable as a Python function via
`substrate-interface`. Install the extra and bring up a dev node:

```bash
uv sync --extra dev --extra chain
docker compose up -d --build node      # first build is slow (full node compile)
```

Run just the bridge integration test:

```bash
uv run pytest tests/test_bridge.py -v
```

Or run the interactive demo, which prints every step + balance/credit deltas:

```bash
uv run python scripts/run_chain_demo.py
```

Real MNIST FL on-chain demo (trains, submits CIDs, validator scores from real
losses, then pulls the winning model back and prints test accuracy):

```bash
uv run python scripts/run_chain_demo_mnist.py
```

What it does:

1. Loads MNIST (defaults: 600 + 120 training samples, 1000 test) via
   `pofl.data.mnist.load_mnist_tensors`.
2. Trains a `TinyMLP` per pool — pool A on 600 samples, pool B on 120
   (deliberately skewed so A normally wins).
3. Serializes each pool's weights and stores them in a local content-addressed
   dict keyed by raw SHA-256 (32 bytes — fits the pallet's 64-byte CID bound).
4. Validator evaluates real test-set loss for each pool, converted to
   `loss × 1e9` to match the pallet's `score_q9` field.
5. Full chain flow via the bridge: `publish_task` → both pool captains
   `lock_participation` → `register_pool_members` / `register_pool_wallet` for
   both pools → `submit_fl_model` (per pool) → wait for `release_block` →
   `reveal_test_key` → `Sudo.set_validator_credit` → `report_model_score`
   (per pool) → `validator_vote_winner` → `finalize_task`.
6. After finalize: reads the on-chain `TaskSettled.winner` event, queries
   `FlSubmissions[task_id, winner_pool]` for the weights CID, fetches the bytes
   from the local store, loads them into a model, and prints **test accuracy of
   the winning model** plus the losing pool's accuracy for context.

Only the pre-funded dev accounts are used (`//Alice` as requester+sudo,
`//Bob` as pool A captain, `//Alice//stash` as pool B captain, `//Bob//stash`
as validator), so no extra balance transfers are needed.

Useful flags: `--pool-a-samples`, `--pool-b-samples`, `--test-samples`,
`--epochs`, `--hidden`, `--lr`, `--seed`, `--mnist-root`,
`--url ws://host:9944`.

Sample tail of a successful run:

```text
=== 7. finalize_task ===
  [finalize_task              ] ok
      event PfPofl.TaskSettled
  on-chain winner pool = b'pool-mnist-A'
  winner pool wallet gained: 11.0000 UNIT

=== 8. Pull winning model from chain → IPFS → evaluate accuracy ===
  on-chain round = 1
  fetched weights (54xxx bytes) for cid 6a3f9c...
  WINNING MODEL TEST ACCURACY = 86.40%  (on 1000 samples)
  (losing pool b'pool-mnist-B' accuracy = 63.20%)
```

It skips automatically when no node is reachable on `ws://127.0.0.1:9944` (override
with `POFL_NODE_URL`). The test exercises the full lifecycle: `publish_task` →
`lock_participation` → `register_pool_members`/`register_pool_wallet` →
`submit_fl_model` → `reveal_test_key` → `Sudo.set_validator_credit` →
`report_model_score` → `validator_vote_winner` → `finalize_task`, plus a
secondary check on `submit_vrf_ticket` and `agreement_advance_round`.

Programmatic use:

```python
from pofl.chain import ChainClient, extrinsics as ex

with ChainClient("ws://127.0.0.1:9944") as c:
    alice = c.keypair("//Alice")
    res = ex.publish_task(c, alice, reward=10**13, hosting_fee=10**12,
                          participation_deposit=10**12,
                          training_deadline=200, release_block=20,
                          initial_model_cid=b"QmInit", test_ciphertext_cid=b"QmCt",
                          test_commitment=b"\x00" * 32)
    print(res.success, res.event("PfPofl", "TaskPublished"))
```

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
