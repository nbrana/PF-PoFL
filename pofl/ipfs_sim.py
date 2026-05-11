from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


def _cid_from_bytes(content: bytes) -> str:
    return "sha256-" + hashlib.sha256(content).hexdigest()


class IPFSimulator:
    """Content-addressed store mimicking IPFS CIDs (SHA-256 digest)."""

    def __init__(self, storage_dir: Optional[str | Path] = None) -> None:
        self._mem: dict[str, bytes] = {}
        self.storage_dir: Optional[Path] = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

    def put(self, content: bytes) -> str:
        cid = _cid_from_bytes(content)
        self._mem[cid] = content
        if self.storage_dir:
            path = self.storage_dir / cid.replace("sha256-", "")
            path.write_bytes(content)
        return cid

    def get(self, cid: str) -> bytes:
        if cid in self._mem:
            return self._mem[cid]
        if self.storage_dir:
            path = self.storage_dir / cid.replace("sha256-", "")
            if path.is_file():
                data = path.read_bytes()
                self._mem[cid] = data
                return data
        raise KeyError(f"CID not found: {cid}")

    def has(self, cid: str) -> bool:
        try:
            self.get(cid)
            return True
        except KeyError:
            return False
