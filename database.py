"""SQLite persistence for chat sessions, messages, and uploaded file metadata."""
import json
import os
import sqlite3
from typing import Any

from utils import utc_iso

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_data.db")


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL DEFAULT '',
                persona     TEXT NOT NULL DEFAULT 'default',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role            TEXT NOT NULL CHECK(role IN ('user','assistant','system','tool')),
                content         TEXT NOT NULL DEFAULT '',
                images          TEXT,
                thinking        TEXT,
                tool_calls      TEXT,
                search_data     TEXT,
                tps             TEXT,
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);

            CREATE TABLE IF NOT EXISTS uploaded_files (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                filename        TEXT NOT NULL,
                original_name   TEXT NOT NULL,
                mime_type       TEXT NOT NULL,
                size            INTEGER NOT NULL,
                url_path        TEXT NOT NULL,
                message_id      INTEGER REFERENCES chat_messages(id) ON DELETE SET NULL,
                created_at      TEXT NOT NULL
            );
        """)


# ── Session CRUD ──────────────────────────────

def create_session(session_id: str, name: str = "", persona: str = "default") -> dict:
    now = utc_iso()
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO chat_sessions(id, name, persona, created_at, updated_at) "
            "VALUES(?, ?, ?, COALESCE((SELECT created_at FROM chat_sessions WHERE id=?), ?), ?)",
            (session_id, name, persona, session_id, now, now),
        )
    return {"id": session_id, "name": name, "persona": persona}


def update_session(session_id: str, name: str | None = None, persona: str | None = None) -> bool:
    fields: list[str] = []
    values: list[Any] = []
    if name is not None:
        fields.append("name=?")
        values.append(name)
    if persona is not None:
        fields.append("persona=?")
        values.append(persona)
    if not fields:
        return False
    fields.append("updated_at=?")
    values.append(utc_iso())
    values.append(session_id)
    with get_db() as db:
        cur = db.execute(f"UPDATE chat_sessions SET {', '.join(fields)} WHERE id=?", values)
        return cur.rowcount > 0


def list_sessions() -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM chat_messages WHERE session_id=s.id) AS message_count "
            "FROM chat_sessions s ORDER BY s.updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    with get_db() as db:
        cur = db.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))
        return cur.rowcount > 0


# ── Messages ──────────────────────────────────

def save_messages(session_id: str, messages: list[dict]) -> int:
    """Batch-insert messages. Returns count inserted."""
    now = utc_iso()
    count = 0
    with get_db() as db:
        db.execute("BEGIN")
        for m in messages:
            db.execute(
                "INSERT INTO chat_messages(session_id, role, content, images, thinking, tool_calls, search_data, tps, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    m.get("role", "user"),
                    m.get("content", ""),
                    json.dumps(m.get("images")) if m.get("images") else None,
                    m.get("thinking"),
                    json.dumps(m.get("tool_calls")) if m.get("tool_calls") else None,
                    json.dumps(m.get("search_data")) if m.get("search_data") else None,
                    m.get("tps"),
                    now,
                ),
            )
            count += 1
        db.execute("COMMIT")
    return count


def get_messages(session_id: str) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    result: list[dict] = []
    for r in rows:
        d = dict(r)
        for field in ("images", "tool_calls", "search_data"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = None
        result.append(d)
    return result


def delete_messages(session_id: str) -> int:
    with get_db() as db:
        cur = db.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        return cur.rowcount


# ── Uploaded files ────────────────────────────

def save_file_meta(
    filename: str,
    original_name: str,
    mime_type: str,
    size: int,
    url_path: str,
    message_id: int | None = None,
) -> int:
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO uploaded_files(filename, original_name, mime_type, size, url_path, message_id, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (filename, original_name, mime_type, size, url_path, message_id, utc_iso()),
        )
        return cur.lastrowid


def get_files_for_message(message_id: int) -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM uploaded_files WHERE message_id=? ORDER BY id ASC",
            (message_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── User API providers (stored server-side, never in browser) ──

def init_providers_table() -> None:
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS user_api_providers (
                id      TEXT PRIMARY KEY,
                name    TEXT NOT NULL DEFAULT '',
                base_url TEXT NOT NULL DEFAULT '',
                api_key TEXT NOT NULL DEFAULT ''
            )
        """)


def save_providers(providers: list[dict]) -> None:
    with get_db() as db:
        db.execute("DELETE FROM user_api_providers")
        for p in providers:
            db.execute(
                "INSERT INTO user_api_providers(id, name, base_url, api_key) VALUES(?,?,?,?)",
                (p.get("id", ""), p.get("name", ""), p.get("baseUrl", ""), p.get("apiKey", "")),
            )


def list_providers() -> list[dict]:
    with get_db() as db:
        rows = db.execute("SELECT * FROM user_api_providers").fetchall()
    return [dict(r) for r in rows]


# ── Auto-init ─────────────────────────────────
init_db()
init_providers_table()
