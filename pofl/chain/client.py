"""Substrate connection + signing helpers for the PF-PoFL bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from substrateinterface import Keypair, SubstrateInterface
    from substrateinterface.exceptions import SubstrateRequestException
except ImportError as exc:
    raise ImportError(
        "substrate-interface is required for pofl.chain. "
        "Install with: uv sync --extra chain  (or pip install substrate-interface)"
    ) from exc


PALLET = "PfPofl"
SUDO_PALLET = "Sudo"
DEFAULT_URL = "ws://127.0.0.1:9944"


@dataclass
class ExtrinsicResult:
    """Outcome of an extrinsic submission."""

    success: bool
    block_hash: str
    extrinsic_hash: str
    events: list[dict[str, Any]]
    error: str | None = None

    def event(self, pallet: str, name: str) -> dict[str, Any] | None:
        for ev in self.events:
            if ev["pallet"] == pallet and ev["event"] == name:
                return ev
        return None


class ChainClient:
    """Thin wrapper over `SubstrateInterface` tailored to pallet-pf-pofl."""

    def __init__(self, url: str = DEFAULT_URL):
        self.url = url
        self._si = SubstrateInterface(url=url)

    def close(self) -> None:
        self._si.close()

    def __enter__(self) -> ChainClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def keypair(uri: str) -> Keypair:
        """Derive a Keypair from a SURI (e.g. `//Alice`)."""
        return Keypair.create_from_uri(uri)

    @property
    def substrate(self) -> SubstrateInterface:
        return self._si

    def compose(self, function: str, params: dict[str, Any]):
        return self._si.compose_call(
            call_module=PALLET, call_function=function, call_params=params
        )

    def submit(
        self,
        function: str,
        params: dict[str, Any],
        signer: Keypair,
        wait_for_inclusion: bool = True,
    ) -> ExtrinsicResult:
        return self._sign_and_send(self.compose(function, params), signer, wait_for_inclusion)

    def submit_sudo(
        self,
        function: str,
        params: dict[str, Any],
        signer: Keypair,
        wait_for_inclusion: bool = True,
    ) -> ExtrinsicResult:
        """Wrap a call in `Sudo.sudo` for root-origin dispatchables."""
        inner = self.compose(function, params)
        outer = self._si.compose_call(
            call_module=SUDO_PALLET, call_function="sudo", call_params={"call": inner.value}
        )
        return self._sign_and_send(outer, signer, wait_for_inclusion)

    def _sign_and_send(self, call, signer: Keypair, wait_for_inclusion: bool) -> ExtrinsicResult:
        extrinsic = self._si.create_signed_extrinsic(call=call, keypair=signer)
        try:
            receipt = self._si.submit_extrinsic(
                extrinsic, wait_for_inclusion=wait_for_inclusion
            )
        except SubstrateRequestException as exc:
            return ExtrinsicResult(
                success=False,
                block_hash="",
                extrinsic_hash="",
                events=[],
                error=str(exc),
            )

        events = _normalize_events(receipt.triggered_events) if wait_for_inclusion else []
        err = None
        if wait_for_inclusion and not receipt.is_success:
            err_obj = receipt.error_message
            err = err_obj.get("name") if isinstance(err_obj, dict) else str(err_obj)
        return ExtrinsicResult(
            success=bool(wait_for_inclusion and receipt.is_success),
            block_hash=str(receipt.block_hash or ""),
            extrinsic_hash=str(receipt.extrinsic_hash or ""),
            events=events,
            error=err,
        )

    def query(self, storage: str, params: list[Any] | None = None) -> Any:
        result = self._si.query(module=PALLET, storage_function=storage, params=params or [])
        return result.value if result is not None else None

    def free_balance(self, address: str) -> int:
        info = self._si.query("System", "Account", [address])
        return int(info.value["data"]["free"]) if info is not None else 0

    def transfer(self, signer: Keypair, dest: str, amount: int) -> ExtrinsicResult:
        """Convenience: `Balances.transfer_keep_alive` from `signer` to `dest`."""
        call = self._si.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={"dest": dest, "value": amount},
        )
        return self._sign_and_send(call, signer, wait_for_inclusion=True)

    def ensure_funded(self, funder: Keypair, dest: str, minimum: int) -> None:
        """Top up `dest` from `funder` so it has at least `minimum` free balance."""
        have = self.free_balance(dest)
        if have >= minimum:
            return
        res = self.transfer(funder, dest, minimum - have + 1)
        if not res.success:
            raise RuntimeError(f"failed to fund {dest}: {res.error}")

    def block_number(self) -> int:
        head = self._si.get_block_number(self._si.get_chain_head())
        return int(head)

    def wait_until_block(self, target: int, timeout_blocks: int = 60) -> int:
        """Block (polling) until chain head >= target. Returns the head reached."""
        import time

        start_head = self.block_number()
        for _ in range(timeout_blocks * 2):
            head = self.block_number()
            if head >= target:
                return head
            time.sleep(3)
        raise TimeoutError(
            f"chain did not reach block {target} (started {start_head}, "
            f"last {self.block_number()})"
        )


def _normalize_events(triggered) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in triggered or []:
        # substrate-interface wraps events differently across versions; normalize.
        inner = ev.value if hasattr(ev, "value") else ev
        ev_root = inner.get("event", inner) if isinstance(inner, dict) else {}
        out.append(
            {
                "pallet": ev_root.get("module_id") or ev_root.get("pallet") or "",
                "event": ev_root.get("event_id") or ev_root.get("event") or "",
                "attributes": ev_root.get("attributes") or ev_root.get("params") or {},
            }
        )
    return out
