"""MVP test-data holdback: XOR with a key; chain commits to key hash until reveal block.

Security note (see spec.md): symmetric key revealed on-chain is PoC-only; production needs
threshold / time-lock release.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass


@dataclass
class HoldbackBundle:
    ciphertext: bytes
    key: bytes
    key_commitment: str


def xor_bytes(data: bytes, key: bytes) -> bytes:
    out = bytearray(len(data))
    for i, b in enumerate(data):
        out[i] = b ^ key[i % len(key)]
    return bytes(out)


def prepare_test_holdback(plaintext: bytes, key_len: int = 32) -> HoldbackBundle:
    key = secrets.token_bytes(key_len)
    ct = xor_bytes(plaintext, key)
    commitment = hashlib.sha256(key).hexdigest()
    return HoldbackBundle(ciphertext=ct, key=key, key_commitment=commitment)


def reveal_key(key: bytes, expected_commitment_hex: str) -> bool:
    return hashlib.sha256(key).hexdigest() == expected_commitment_hex


def decrypt_test(ciphertext: bytes, key: bytes) -> bytes:
    return xor_bytes(ciphertext, key)
