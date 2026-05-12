"""Python bridge to the `pallet-pf-pofl` Substrate runtime.

Lightweight wrappers around `substrate-interface` so the existing PoFL role
code can submit on-chain extrinsics matching the FRAME pallet's dispatchables.
"""

from pofl.chain import extrinsics
from pofl.chain.client import ChainClient

__all__ = ["ChainClient", "extrinsics"]
