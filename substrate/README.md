# Substrate chain (PF-PoFL)

This directory vendors the `[polkadot-sdk-solochain-template](https://github.com/paritytech/polkadot-sdk-solochain-template)` as `[chain/](chain)`.

## Layout

- `[chain/pallets/pf_pofl/](chain/pallets/pf_pofl)` — single FRAME pallet combining **task publication**, **participation deposits**, **FL model CIDs**, **test-key reveal (holdback)**, **VRF ticket stub**, **agreement round counter**, **validator winner votes (credit-weighted tally)**, and **finalize** (winner payout + hosting-fee split). Maps conceptually to *pfl-tasks*, *pfl-fl*, *pfl-settlement*, and *pfl-agreement* from the architecture spec.
- `[chain/runtime/](chain/runtime)` — Aura + Grandpa solochain runtime wiring `PfPofl` at pallet index 7.
- `[spec.md](../spec.md)` — paper-aligned specification and limitations.

## Docker (g++ / RocksDB / full node build)

From the **repository root**:

```bash
docker build -t pf-pofl-node .
docker run --rm pf-pofl-node -- --help
```

The image uses a **bookworm** Rust builder with `clang`, `cmake`, `g++`, Snappy/LZ4/Zstd/Zlib dev packages so `**librocksdb-sys`** can compile. The runtime stage is **debian:bookworm-slim** plus OpenSSL certs (the node binary links required libs statically where possible).

See the root `[Dockerfile](../Dockerfile)` for details.

## Build

From `substrate/chain`:

```bash
cargo check -p pallet-pf-pofl
cargo check -p solochain-template-runtime
```

Building the **node** (`solochain-template-node`) compiles native RocksDB and may require a full C++ toolchain; if that fails in CI or minimal images, rely on runtime checks above or use Polkadot SDK’s WASM-only workflows.

## Genesis

The PF-PoFL pot account (`PalletId(*b"pf_pofl!")`) receives an extra endowment in `[runtime/src/genesis_config_presets.rs](chain/runtime/src/genesis_config_presets.rs)` so escrow transfers succeed in dev presets.

## Python bridge

Training and IPFS remain in the top-level `[pofl/](../pofl)` package; workers should submit extrinsics matching `pallet-pf-pofl` dispatchables. Deterministic scoring helpers live in `[pofl/ocw_scoring.py](../pofl/ocw_scoring.py)`.