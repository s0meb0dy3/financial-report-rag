import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SESSION_DB_PATH = "data/sessions.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_list(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _json_object(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


@dataclass
class SessionSummary:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class SessionTurn:
    id: int
    session_id: str
    user_content: str
    assistant_content: str
    reasoning_content: str
    citations: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    usage: dict[str, Any] | None
    created_at: str


class SQLiteSessionStore:
    """SQLite chat history store.

    The store only owns chat sessions and turns. Model calls, tool execution, and
    UI state stay outside the persistence layer so the app remains easy to trim.
    """

    @classmethod
    def from_env(cls) -> "SQLiteSessionStore":
        return cls(os.environ.get("SESSION_DB_PATH", DEFAULT_SESSION_DB_PATH))

    def __init__(self, db_path: str | Path = DEFAULT_SESSION_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_content TEXT NOT NULL,
                    assistant_content TEXT NOT NULL,
                    reasoning_content TEXT NOT NULL DEFAULT '',
                    citations_json TEXT NOT NULL,
                    tool_results_json TEXT NOT NULL DEFAULT '[]',
                    usage_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )
        self._ensure_turn_metadata_columns()

    def _ensure_turn_metadata_columns(self) -> None:
        """Keep databases created by older versions readable."""
        with self._connect() as connection:
            rows = connection.execute("PRAGMA table_info(session_turns)").fetchall()
            columns = {row["name"] for row in rows}
            if "reasoning_content" not in columns:
                connection.execute(
                    "ALTER TABLE session_turns ADD COLUMN reasoning_content TEXT NOT NULL DEFAULT ''"
                )
            if "usage_json" not in columns:
                connection.execute("ALTER TABLE session_turns ADD COLUMN usage_json TEXT")
            if "tool_results_json" not in columns:
                connection.execute(
                    "ALTER TABLE session_turns ADD COLUMN tool_results_json TEXT NOT NULL DEFAULT '[]'"
                )

    def ensure_session(self, session_id: str, *, title: str | None = None) -> SessionSummary:
        existing = self.get_session(session_id)
        if existing is not None:
            return existing

        now = _now()
        resolved_title = (title or "新对话").strip() or "新对话"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, resolved_title, now, now),
            )
        return SessionSummary(id=session_id, title=resolved_title, created_at=now, updated_at=now)

    def create_session(self, session_id: str, *, title: str = "新对话") -> SessionSummary:
        return self.ensure_session(session_id, title=title)

    def list_sessions(self) -> list[SessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def get_session(self, session_id: str) -> SessionSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return self._row_to_summary(row) if row else None

    def update_session(self, session_id: str, *, title: str | None = None) -> SessionSummary:
        session = self.get_session(session_id)
        if session is None:
            return self.ensure_session(session_id, title=title)

        resolved_title = session.title if title is None else title.strip() or session.title
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (resolved_title, now, session_id),
            )
        return SessionSummary(
            id=session.id,
            title=resolved_title,
            created_at=session.created_at,
            updated_at=now,
        )

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def record_turn(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        citations: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
        reasoning_content: str = "",
        usage: dict[str, Any] | None = None,
    ) -> SessionTurn:
        session = self.ensure_session(session_id, title=user_content[:18] or "新对话")
        if session.title == "新对话" and user_content.strip():
            self.update_session(session_id, title=user_content[:18])

        now = _now()
        insert_columns = [
            "session_id",
            "user_content",
            "assistant_content",
            "reasoning_content",
            "citations_json",
            "tool_results_json",
            "usage_json",
            "created_at",
        ]
        values: list[Any] = [
            session_id,
            user_content,
            assistant_content,
            reasoning_content,
            json.dumps(citations, ensure_ascii=False),
            json.dumps(tool_results or [], ensure_ascii=False),
            json.dumps(usage, ensure_ascii=False) if usage is not None else None,
            now,
        ]

        placeholders = ", ".join("?" for _ in insert_columns)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                INSERT INTO session_turns ({", ".join(insert_columns)})
                VALUES ({placeholders})
                """,
                values,
            )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            turn_id = int(cursor.lastrowid)
        return SessionTurn(
            id=turn_id,
            session_id=session_id,
            user_content=user_content,
            assistant_content=assistant_content,
            reasoning_content=reasoning_content,
            citations=citations,
            tool_results=tool_results or [],
            usage=usage,
            created_at=now,
        )

    def list_turns(self, session_id: str) -> list[SessionTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, user_content, assistant_content,
                       reasoning_content, citations_json, tool_results_json, usage_json, created_at
                FROM session_turns
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            SessionTurn(
                id=int(row["id"]),
                session_id=row["session_id"],
                user_content=row["user_content"],
                assistant_content=row["assistant_content"],
                reasoning_content=row["reasoning_content"] or "",
                citations=_json_list(row["citations_json"]),
                tool_results=_json_list(row["tool_results_json"]),
                usage=_json_object(row["usage_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> SessionSummary:
        return SessionSummary(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
