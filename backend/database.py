import json
import os
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

DB_PATH = os.path.join(os.path.dirname(__file__), "atlas.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL DEFAULT '{}',
                calendar TEXT NOT NULL DEFAULT '{}',
                tasks TEXT NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS message_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                covers_up_to_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tool_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                run_id TEXT,
                agent TEXT NOT NULL,
                tool TEXT NOT NULL,
                args TEXT,
                status TEXT NOT NULL,            -- allowed | denied | error
                result TEXT,
                latency_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migration for DBs created before run_id existed
        try:
            await db.execute("ALTER TABLE tool_audit ADD COLUMN run_id TEXT")
        except Exception:
            pass
        await db.commit()


# ── Users ──────────────────────────────────────────────────────────────────────

async def create_user(user_id: str, email: str, password_hash: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
            (user_id, email, password_hash),
        )
        await db.commit()


async def get_user_by_email(email: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?", (email,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def get_user_by_id(user_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, email FROM users WHERE id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


# ── User State (profile / calendar / tasks) ───────────────────────────────────

async def get_user_state(user_id: str) -> Optional[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT profile, calendar, tasks FROM user_state WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "profile": json.loads(row["profile"]),
                "calendar": json.loads(row["calendar"]),
                "tasks": json.loads(row["tasks"]),
            }


async def upsert_user_state(user_id: str, profile: Dict, calendar: Dict, tasks: Dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_state (user_id, profile, calendar, tasks, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                profile = excluded.profile,
                calendar = excluded.calendar,
                tasks = excluded.tasks,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, json.dumps(profile), json.dumps(calendar), json.dumps(tasks)),
        )
        await db.commit()


async def update_state_field(user_id: str, field: str, value: Dict):
    """Update a single JSON field (profile/calendar/tasks) in user_state."""
    allowed = {"profile", "calendar", "tasks"}
    if field not in allowed:
        raise ValueError(f"Invalid field: {field}")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE user_state SET {field} = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
            (json.dumps(value), user_id),
        )
        await db.commit()


# ── Messages ──────────────────────────────────────────────────────────────────

async def append_message(user_id: str, role: str, content: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        await db.commit()
        return cursor.lastrowid


async def get_all_messages(user_id: str) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, role, content FROM messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def delete_messages_up_to(user_id: str, max_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM messages WHERE user_id = ? AND id <= ?",
            (user_id, max_id),
        )
        await db.commit()


async def clear_messages(user_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM message_summaries WHERE user_id = ?", (user_id,))
        await db.commit()


# ── Message Summaries ─────────────────────────────────────────────────────────

async def save_summary(user_id: str, summary: str, covers_up_to_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO message_summaries (user_id, summary, covers_up_to_id) VALUES (?, ?, ?)",
            (user_id, summary, covers_up_to_id),
        )
        await db.commit()


async def get_latest_summary(user_id: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT summary FROM message_summaries WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


# ── Tool Audit (RBAC + traceability) ──────────────────────────────────────────

async def log_tool_call(
    user_id: str,
    agent: str,
    tool: str,
    args: Dict,
    status: str,
    result: str,
    latency_ms: int,
    run_id: str = "",
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO tool_audit (user_id, run_id, agent, tool, args, status, result, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                run_id,
                agent,
                tool,
                json.dumps(args, ensure_ascii=False),
                status,
                (result or "")[:2000],
                latency_ms,
            ),
        )
        await db.commit()


async def find_successful_tool_call(run_id: str, tool: str, args: Dict) -> Optional[str]:
    """Idempotency lookup: has this exact tool+args already succeeded in this run?

    Returns the prior result if so (so a resumed run can replay it without
    re-executing a side-effecting tool), else None.
    """
    if not run_id:
        return None
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT result FROM tool_audit
            WHERE run_id = ? AND tool = ? AND args = ? AND status = 'allowed'
            ORDER BY id ASC LIMIT 1
            """,
            (run_id, tool, json.dumps(args, ensure_ascii=False)),
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def get_tool_audit(user_id: str, limit: int = 100) -> List[Dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT agent, tool, args, status, result, latency_ms, created_at
            FROM tool_audit WHERE user_id = ? ORDER BY id DESC LIMIT ?
            """,
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
