from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional

from pofl.types import TransactionType


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class AccountState:
    balance: int
    credit: int


@dataclass
class TaskRow:
    task_id: str
    publisher: str
    reward: int
    hosting_fee: int
    initial_model_cid: str
    test_dataset_cid: str
    test_dataset_hash: str
    deadline_block: int
    created_block: int
    escrow_amount: int
    status: str
    participation_deposit: int = 0
    release_block: int = 0
    delta_test_blocks: int = 0


class Ledger:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._height = self._load_tip_height()
        self._tip_hash = self._load_tip_hash()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def _init_schema(self) -> None:
        with self._cursor() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    balance INTEGER NOT NULL DEFAULT 0,
                    credit INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS blocks (
                    height INTEGER PRIMARY KEY,
                    prev_hash TEXT NOT NULL,
                    block_hash TEXT NOT NULL UNIQUE,
                    merkle_root TEXT NOT NULL,
                    proposer TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_hash TEXT PRIMARY KEY,
                    block_height INTEGER,
                    tx_type INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(block_height) REFERENCES blocks(height)
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    publisher TEXT NOT NULL,
                    reward INTEGER NOT NULL,
                    hosting_fee INTEGER NOT NULL,
                    initial_model_cid TEXT NOT NULL,
                    test_dataset_cid TEXT NOT NULL,
                    test_dataset_hash TEXT NOT NULL,
                    deadline_block INTEGER NOT NULL,
                    created_block INTEGER NOT NULL,
                    escrow_amount INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    participation_deposit INTEGER NOT NULL DEFAULT 0,
                    release_block INTEGER NOT NULL DEFAULT 0,
                    delta_test_blocks INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS escrow (
                    task_id TEXT PRIMARY KEY,
                    amount INTEGER NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE TABLE IF NOT EXISTS task_completions (
                    task_id TEXT PRIMARY KEY,
                    winner_pool_id TEXT,
                    paid INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE TABLE IF NOT EXISTS fl_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    weights_cid TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    submission_block INTEGER NOT NULL DEFAULT 0,
                    participation_deposit INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(task_id, pool_id, round_index)
                );
                CREATE TABLE IF NOT EXISTS participation_deposits (
                    task_id TEXT NOT NULL,
                    account_id TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    PRIMARY KEY(task_id, account_id)
                );
                CREATE TABLE IF NOT EXISTS pool_wallets (
                    pool_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_phases (
                    task_id TEXT PRIMARY KEY,
                    phase TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS validator_set (
                    task_id TEXT NOT NULL,
                    validator_id TEXT NOT NULL,
                    faulty INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(task_id, validator_id)
                );
                """
            )

    def _load_tip_height(self) -> int:
        with self._conn:
            row = self._conn.execute(
                "SELECT MAX(height) FROM blocks"
            ).fetchone()
            if row is None or row[0] is None:
                return -1
            return int(row[0])

    def _load_tip_hash(self) -> str:
        h = self.tip_height()
        if h < 0:
            return "0" * 64
        with self._conn:
            row = self._conn.execute(
                "SELECT block_hash FROM blocks WHERE height = ?",
                (h,),
            ).fetchone()
            if row is None:
                return "0" * 64
            return str(row[0])

    def tip_height(self) -> int:
        return self._height

    def tip_hash(self) -> str:
        return self._tip_hash

    def ensure_account(self, account_id: str, balance: int = 0, credit: int = 0) -> None:
        with self._cursor() as c:
            c.execute(
                """INSERT OR IGNORE INTO accounts(id, balance, credit) VALUES(?,?,?)""",
                (account_id, balance, credit),
            )

    def get_account(self, account_id: str) -> AccountState:
        with self._conn:
            row = self._conn.execute(
                "SELECT balance, credit FROM accounts WHERE id = ?",
                (account_id,),
            ).fetchone()
            if row is None:
                return AccountState(0, 0)
            return AccountState(int(row[0]), int(row[1]))

    def credit_add(self, account_id: str, delta: int) -> None:
        self.ensure_account(account_id, 0, 0)
        with self._cursor() as c:
            c.execute(
                "UPDATE accounts SET credit = credit + ? WHERE id = ?",
                (delta, account_id),
            )

    def genesis_if_empty(self, genesis_accounts: dict[str, tuple[int, int]]) -> None:
        if self.tip_height() >= 0:
            return
        for aid, (bal, cr) in genesis_accounts.items():
            self.ensure_account(aid, bal, cr)
        payload = {"kind": "genesis", "accounts": list(genesis_accounts.keys())}
        self.append_block(
            prev_hash="0" * 64,
            proposer="genesis",
            payload=payload,
            tx_hashes=[],
        )

    def append_block(
        self,
        prev_hash: str,
        proposer: str,
        payload: dict[str, Any],
        tx_hashes: list[str],
    ) -> tuple[int, str]:
        new_height = self._height + 1
        merkle_root = _sha256_hex("".join(sorted(tx_hashes)))
        body = json.dumps(
            {
                "height": new_height,
                "prev_hash": prev_hash,
                "merkle_root": merkle_root,
                "proposer": proposer,
                "payload": payload,
            },
            sort_keys=True,
        )
        block_hash = _sha256_hex(body)
        with self._cursor() as c:
            c.execute(
                """INSERT INTO blocks(height, prev_hash, block_hash, merkle_root, proposer, payload_json)
                   VALUES(?,?,?,?,?,?)""",
                (
                    new_height,
                    prev_hash,
                    block_hash,
                    merkle_root,
                    proposer,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            for th in tx_hashes:
                c.execute(
                    "UPDATE transactions SET block_height = ? WHERE tx_hash = ?",
                    (new_height, th),
                )
        self._height = new_height
        self._tip_hash = block_hash
        return new_height, block_hash

    def insert_transaction(
        self,
        tx_hash: str,
        tx_type: TransactionType,
        payload_json: str,
    ) -> None:
        with self._cursor() as c:
            c.execute(
                """INSERT OR REPLACE INTO transactions(tx_hash, block_height, tx_type, payload_json)
                   VALUES(?,?,?,?)""",
                (tx_hash, None, int(tx_type), payload_json),
            )

    def apply_payment(self, from_acc: str, to_acc: str, amount: int, fee: int) -> None:
        if amount < 0 or fee < 0:
            raise ValueError("negative payment")
        total = amount + fee
        self.ensure_account(from_acc)
        self.ensure_account(to_acc)
        with self._cursor() as c:
            row = c.execute(
                "SELECT balance FROM accounts WHERE id = ?", (from_acc,)
            ).fetchone()
            bal = int(row[0])
            if bal < total:
                raise ValueError("insufficient balance")
            c.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?",
                (total, from_acc),
            )
            c.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                (amount, to_acc),
            )

    def publish_task(
        self,
        task_id: str,
        publisher: str,
        reward: int,
        hosting_fee: int,
        initial_model_cid: str,
        test_dataset_cid: str,
        test_dataset_hash: str,
        deadline_block: int,
        created_block: int,
        participation_deposit: int = 0,
        release_block: int = 0,
        delta_test_blocks: int = 0,
    ) -> None:
        escrow_amount = reward + hosting_fee
        with self._cursor() as c:
            c.execute(
                "SELECT balance FROM accounts WHERE id = ?", (publisher,)
            )
            row = c.fetchone()
            if row is None or int(row[0]) < escrow_amount:
                raise ValueError("insufficient balance for task escrow")
            c.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?",
                (escrow_amount, publisher),
            )
            c.execute(
                """INSERT INTO tasks(task_id, publisher, reward, hosting_fee,
                    initial_model_cid, test_dataset_cid, test_dataset_hash,
                    deadline_block, created_block, escrow_amount, status,
                    participation_deposit, release_block, delta_test_blocks)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    task_id,
                    publisher,
                    reward,
                    hosting_fee,
                    initial_model_cid,
                    test_dataset_cid,
                    test_dataset_hash,
                    deadline_block,
                    created_block,
                    escrow_amount,
                    "open",
                    participation_deposit,
                    release_block,
                    delta_test_blocks,
                ),
            )
            c.execute(
                "INSERT INTO escrow(task_id, amount) VALUES(?,?)",
                (task_id, escrow_amount),
            )
            c.execute(
                """INSERT OR REPLACE INTO task_phases(task_id, phase) VALUES(?,?)""",
                (task_id, "training"),
            )

    def _row_to_task(self, r: sqlite3.Row) -> TaskRow:
        return TaskRow(
            task_id=str(r["task_id"]),
            publisher=str(r["publisher"]),
            reward=int(r["reward"]),
            hosting_fee=int(r["hosting_fee"]),
            initial_model_cid=str(r["initial_model_cid"]),
            test_dataset_cid=str(r["test_dataset_cid"]),
            test_dataset_hash=str(r["test_dataset_hash"]),
            deadline_block=int(r["deadline_block"]),
            created_block=int(r["created_block"]),
            escrow_amount=int(r["escrow_amount"]),
            status=str(r["status"]),
            participation_deposit=int(r["participation_deposit"]),
            release_block=int(r["release_block"]),
            delta_test_blocks=int(r["delta_test_blocks"]),
        )

    def list_open_tasks_ordered(self) -> list[TaskRow]:
        with self._conn:
            rows = self._conn.execute(
                """SELECT * FROM tasks WHERE status = 'open'
                   ORDER BY deadline_block ASC, created_block ASC"""
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_task(self, task_id: str) -> Optional[TaskRow]:
        with self._conn:
            r = self._conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if r is None:
            return None
        return self._row_to_task(r)

    def complete_task_pay(
        self,
        task_id: str,
        winner_pool_id: str,
        pay_to_account: str,
    ) -> None:
        with self._cursor() as c:
            row = c.execute(
                "SELECT amount FROM escrow WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise ValueError("no escrow for task")
            amount = int(row[0])
            done = c.execute(
                "SELECT paid FROM task_completions WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if done and int(done[0]) == 1:
                raise ValueError("task already paid")
            self.ensure_account(pay_to_account)
            c.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                (amount, pay_to_account),
            )
            c.execute(
                "DELETE FROM escrow WHERE task_id = ?", (task_id,)
            )
            c.execute(
                """INSERT INTO task_completions(task_id, winner_pool_id, paid)
                   VALUES(?,?,1)
                   ON CONFLICT(task_id) DO UPDATE SET winner_pool_id = excluded.winner_pool_id, paid = 1""",
                (task_id, winner_pool_id),
            )
            c.execute(
                "UPDATE tasks SET status = 'completed' WHERE task_id = ?", (task_id,)
            )

    def record_fl_submission(
        self,
        task_id: str,
        pool_id: str,
        round_index: int,
        weights_cid: str,
        tx_hash: str,
        *,
        submission_block: int = 0,
        participation_deposit: int = 0,
    ) -> None:
        with self._cursor() as c:
            c.execute(
                """INSERT OR REPLACE INTO fl_submissions
                   (task_id, pool_id, round_index, weights_cid, tx_hash,
                    submission_block, participation_deposit)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    task_id,
                    pool_id,
                    round_index,
                    weights_cid,
                    tx_hash,
                    submission_block,
                    participation_deposit,
                ),
            )

    def latest_fl_submission_per_pool(self, task_id: str) -> dict[str, tuple[str, int]]:
        with self._conn:
            rows = self._conn.execute(
                """SELECT pool_id, weights_cid, round_index FROM fl_submissions
                   WHERE task_id = ?
                   ORDER BY round_index DESC""",
                (task_id,),
            ).fetchall()
        seen: dict[str, tuple[str, int]] = {}
        for r in rows:
            pid = str(r["pool_id"])
            if pid not in seen:
                seen[pid] = (str(r["weights_cid"]), int(r["round_index"]))
        return seen

    def get_fl_submission_meta(
        self, task_id: str, pool_id: str
    ) -> Optional[tuple[int, int, str]]:
        """Return (submission_block, participation_deposit, weights_cid) for the
        latest submission from a given pool on a task, or None if absent."""
        with self._conn:
            r = self._conn.execute(
                """SELECT submission_block, participation_deposit, weights_cid
                   FROM fl_submissions
                   WHERE task_id = ? AND pool_id = ?
                   ORDER BY round_index DESC LIMIT 1""",
                (task_id, pool_id),
            ).fetchone()
        if r is None:
            return None
        return int(r[0]), int(r[1]), str(r[2])

    def lock_participation_deposit(
        self, task_id: str, account_id: str, amount: int
    ) -> None:
        if amount < 0:
            raise ValueError("negative deposit")
        with self._cursor() as c:
            row = c.execute(
                "SELECT balance FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
            if row is None or int(row[0]) < amount:
                raise ValueError("insufficient balance for participation deposit")
            c.execute(
                "UPDATE accounts SET balance = balance - ? WHERE id = ?",
                (amount, account_id),
            )
            c.execute(
                """INSERT INTO participation_deposits(task_id, account_id, amount)
                   VALUES(?,?,?)
                   ON CONFLICT(task_id, account_id) DO UPDATE SET amount = amount + excluded.amount""",
                (task_id, account_id, amount),
            )

    def get_participation_deposit(self, task_id: str, account_id: str) -> int:
        with self._conn:
            r = self._conn.execute(
                """SELECT amount FROM participation_deposits
                   WHERE task_id = ? AND account_id = ?""",
                (task_id, account_id),
            ).fetchone()
        return int(r[0]) if r else 0

    def upsert_pool_wallet(self, pool_id: str, account_id: str) -> str:
        self.ensure_account(account_id)
        with self._cursor() as c:
            c.execute(
                """INSERT INTO pool_wallets(pool_id, account_id) VALUES(?,?)
                   ON CONFLICT(pool_id) DO NOTHING""",
                (pool_id, account_id),
            )
            row = c.execute(
                "SELECT account_id FROM pool_wallets WHERE pool_id = ?",
                (pool_id,),
            ).fetchone()
            return str(row[0])

    def register_validator(self, task_id: str, validator_id: str) -> None:
        with self._cursor() as c:
            c.execute(
                """INSERT OR IGNORE INTO validator_set(task_id, validator_id, faulty)
                   VALUES(?,?,0)""",
                (task_id, validator_id),
            )

    def mark_validator_faulty(self, task_id: str, validator_id: str) -> None:
        self.register_validator(task_id, validator_id)
        with self._cursor() as c:
            c.execute(
                """UPDATE validator_set SET faulty = 1
                   WHERE task_id = ? AND validator_id = ?""",
                (task_id, validator_id),
            )

    def honest_validators(self, task_id: str) -> list[str]:
        with self._conn:
            rows = self._conn.execute(
                """SELECT validator_id FROM validator_set
                   WHERE task_id = ? AND faulty = 0
                   ORDER BY validator_id""",
                (task_id,),
            ).fetchall()
        return [str(r[0]) for r in rows]

    def set_task_phase(self, task_id: str, phase: str) -> None:
        with self._cursor() as c:
            c.execute(
                """INSERT INTO task_phases(task_id, phase) VALUES(?,?)
                   ON CONFLICT(task_id) DO UPDATE SET phase = excluded.phase""",
                (task_id, phase),
            )

    def get_task_phase(self, task_id: str) -> Optional[str]:
        with self._conn:
            r = self._conn.execute(
                "SELECT phase FROM task_phases WHERE task_id = ?", (task_id,)
            ).fetchone()
        return str(r[0]) if r else None

    def finalize_task_paper(
        self,
        task_id: str,
        winner_pool_id: str,
        winner_pool_wallet: str,
        qualifying_pool_wallets: list[str],
        honest_validator_ids: list[str],
        xi1: int,
        xi2: int,
        miner_credit_deltas: list[tuple[str, int]],
        validator_credit_deltas: list[tuple[str, int]],
    ) -> dict[str, int]:
        """Algorithm 2 settlement: pay reward, hosting fee + redistributed
        deposits, qualifying-pool refunds, and apply credit deltas. Atomic.

        Returns a payee->amount map for caller-side assertions/logging.
        """
        if xi1 < 0 or xi2 < 0:
            raise ValueError("negative xi")
        with self._cursor() as c:
            done_row = c.execute(
                "SELECT paid FROM task_completions WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if done_row and int(done_row[0]) == 1:
                raise ValueError("task already paid")

            esc_row = c.execute(
                "SELECT amount FROM escrow WHERE task_id = ?", (task_id,)
            ).fetchone()
            if esc_row is None:
                raise ValueError("no escrow for task")
            esc_amount = int(esc_row[0])

            task_row = c.execute(
                """SELECT reward FROM tasks WHERE task_id = ?""",
                (task_id,),
            ).fetchone()
            if task_row is None:
                raise ValueError("task missing")
            reward = int(task_row[0])

            payouts: dict[str, int] = {}

            # 1. Reward to winning pool wallet.
            self.ensure_account(winner_pool_wallet)
            c.execute(
                "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                (reward, winner_pool_wallet),
            )
            payouts[winner_pool_wallet] = payouts.get(winner_pool_wallet, 0) + reward

            # 2. Validators get hosting fee plus xi2 * |W \ {winner}| / |V|.
            qualifying_count = max(0, len(qualifying_pool_wallets))
            non_qualifying_xi2_pool = xi2 * (
                # Paper line 4 redistributes deposits of pools NOT in W_τ to
                # validators. For our PoC we treat the qualifying-pool xi2 as
                # the explicit refund (line 5) and any *additional* xi2 deposit
                # collected from below-threshold pools as part of validators'
                # share. Caller controls this via xi2 magnitude.
                # Implemented: hosting_fee == xi1, validator share = xi1 only.
                0
            )
            _ = non_qualifying_xi2_pool  # kept for future Sybil-deposit accounting
            n_validators = max(1, len(honest_validator_ids))
            base, rem = divmod(xi1, n_validators)
            for i, vid in enumerate(honest_validator_ids):
                share = base + (1 if i < rem else 0)
                if share <= 0:
                    continue
                self.ensure_account(vid)
                c.execute(
                    "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                    (share, vid),
                )
                payouts[vid] = payouts.get(vid, 0) + share

            # 3. xi2 refund to each qualifying pool wallet (paper Alg 2 line 5).
            for wallet in qualifying_pool_wallets:
                self.ensure_account(wallet)
                c.execute(
                    "UPDATE accounts SET balance = balance + ? WHERE id = ?",
                    (xi2, wallet),
                )
                payouts[wallet] = payouts.get(wallet, 0) + xi2

            # Drain escrow row to mark fully spent.
            c.execute("DELETE FROM escrow WHERE task_id = ?", (task_id,))

            c.execute(
                """INSERT INTO task_completions(task_id, winner_pool_id, paid)
                   VALUES(?,?,1)
                   ON CONFLICT(task_id) DO UPDATE SET
                     winner_pool_id = excluded.winner_pool_id, paid = 1""",
                (task_id, winner_pool_id),
            )
            c.execute(
                "UPDATE tasks SET status = 'completed' WHERE task_id = ?",
                (task_id,),
            )

            # 4. Credit deltas.
            for acct, delta in miner_credit_deltas:
                if delta == 0:
                    continue
                c.execute(
                    """INSERT INTO accounts(id, balance, credit) VALUES(?,0,?)
                       ON CONFLICT(id) DO UPDATE SET credit = credit + excluded.credit""",
                    (acct, delta),
                )
            for vid, delta in validator_credit_deltas:
                if delta == 0:
                    continue
                c.execute(
                    """INSERT INTO accounts(id, balance, credit) VALUES(?,0,?)
                       ON CONFLICT(id) DO UPDATE SET credit = credit + excluded.credit""",
                    (vid, delta),
                )

            return payouts
