# Build from repository root:
#   docker build -t pf-pofl-node .
#
# Run binary (flags after --):
#   docker run --rm pf-pofl-node -- --help
#
# Interactive shell (toolchain + source in image only if you extend Dockerfile):
#   docker run --rm -it --entrypoint bash pf-pofl-node

FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    curl \
    git \
    libclang-dev \
    liblz4-dev \
    libsnappy-dev \
    libssl-dev \
    libzstd-dev \
    pkg-config \
    protobuf-compiler \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# `substrate-wasm-builder` needs the standard library sources for the WASM runtime.
RUN rustup target add wasm32-unknown-unknown \
    && rustup component add rust-src

WORKDIR /build
COPY substrate/chain/ ./

ENV RUST_BACKTRACE=1
RUN cargo build --locked --release -p solochain-template-node

FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/solochain-template-node /usr/local/bin/

ENTRYPOINT ["/usr/local/bin/solochain-template-node"]
