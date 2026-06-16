"""SQLite-backed LangGraph checkpointer for crash/interrupt recovery.

The official `langgraph-checkpoint-sqlite` package isn't available in this
environment, so this mirrors `InMemorySaver`'s data model (checkpoints / blobs /
writes) but persists every super-step to SQLite. With it the compiled graph
snapshots the full `AcademicState` after each node; a crashed or disconnected
long ReAct run can be resumed from the last completed node using the same
`thread_id`, instead of re-running from scratch.

Sync methods do the real work (blocking sqlite3, fine for a single-process async
app); async methods delegate, exactly like InMemorySaver.
"""
from __future__ import annotations

import random
import sqlite3
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class SqliteCheckpointer(BaseCheckpointSaver[str]):
    def __init__(self, db_path: str, *, serde=None):
        super().__init__(serde=serde)
        self.db_path = db_path
        self._setup()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint BLOB,
                    metadata_type TEXT,
                    metadata BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BLOB,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    blob BLOB,
                    task_path TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
            """)
            conn.commit()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _load_blobs(self, conn, thread_id, checkpoint_ns, versions: ChannelVersions) -> dict:
        values: dict[str, Any] = {}
        for channel, ver in versions.items():
            row = conn.execute(
                "SELECT type, blob FROM checkpoint_blobs "
                "WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?",
                (thread_id, checkpoint_ns, channel, str(ver)),
            ).fetchone()
            if row and row["type"] != "empty":
                values[channel] = self.serde.loads_typed((row["type"], row["blob"]))
        return values

    def _load_writes(self, conn, thread_id, checkpoint_ns, checkpoint_id) -> list:
        rows = conn.execute(
            "SELECT task_id, channel, type, blob FROM checkpoint_writes "
            "WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? ORDER BY idx",
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()
        return [
            (r["task_id"], r["channel"], self.serde.loads_typed((r["type"], r["blob"])))
            for r in rows
        ]

    def _row_to_tuple(self, conn, thread_id, checkpoint_ns, row) -> CheckpointTuple:
        checkpoint_id = row["checkpoint_id"]
        checkpoint: Checkpoint = self.serde.loads_typed((row["type"], row["checkpoint"]))
        metadata = self.serde.loads_typed((row["metadata_type"], row["metadata"]))
        parent = row["parent_checkpoint_id"]
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    conn, thread_id, checkpoint_ns, checkpoint["channel_versions"]
                ),
            },
            metadata=metadata,
            pending_writes=self._load_writes(conn, thread_id, checkpoint_ns, checkpoint_id),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent,
                    }
                }
                if parent
                else None
            ),
        )

    # ── sync core ──────────────────────────────────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self._conn() as conn:
            if checkpoint_id := get_checkpoint_id(config):
                row = conn.execute(
                    "SELECT * FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                    "ORDER BY checkpoint_id DESC LIMIT 1",
                    (thread_id, checkpoint_ns),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_tuple(conn, thread_id, checkpoint_ns, row)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        if config is None:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        before_id = get_checkpoint_id(before) if before else None
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id=? AND checkpoint_ns=? "
                "ORDER BY checkpoint_id DESC",
                (thread_id, checkpoint_ns),
            ).fetchall()
        count = 0
        for row in rows:
            if before_id and row["checkpoint_id"] >= before_id:
                continue
            tup = None
            with self._conn() as conn:
                tup = self._row_to_tuple(conn, thread_id, checkpoint_ns, row)
            if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                continue
            if limit is not None and count >= limit:
                break
            count += 1
            yield tup

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        with self._conn() as conn:
            for channel, ver in new_versions.items():
                if channel in values:
                    btype, bblob = self.serde.dumps_typed(values[channel])
                else:
                    btype, bblob = "empty", b""
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoint_blobs "
                    "(thread_id, checkpoint_ns, channel, version, type, blob) VALUES (?,?,?,?,?,?)",
                    (thread_id, checkpoint_ns, channel, str(ver), btype, bblob),
                )
            ctype, cblob = self.serde.dumps_typed(c)
            mtype, mblob = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    ctype,
                    cblob,
                    mtype,
                    mblob,
                ),
            )
            conn.commit()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        with self._conn() as conn:
            for idx, (channel, value) in enumerate(writes):
                widx = WRITES_IDX_MAP.get(channel, idx)
                wtype, wblob = self.serde.dumps_typed(value)
                # idx >= 0: keep the first write (idempotent replay); special
                # negative channels always overwrite.
                verb = "INSERT OR IGNORE" if widx >= 0 else "INSERT OR REPLACE"
                conn.execute(
                    f"{verb} INTO checkpoint_writes "
                    "(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, widx, channel, wtype, wblob, task_path),
                )
            conn.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._conn() as conn:
            for t in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                conn.execute(f"DELETE FROM {t} WHERE thread_id=?", (thread_id,))
            conn.commit()

    def get_next_version(self, current: Optional[str], channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        return f"{next_v:032}.{random.random():016}"

    # ── async delegation ───────────────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)
