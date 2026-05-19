import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SESSION_DB_PATH = "data/sessions.sqlite3"
_UNSET = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_doc_ids(
    doc_ids: list[str] | None = None,
    *,
    fallback_doc_id: str | None = None,
) -> list[str]:
    candidates = doc_ids if doc_ids is not None else ([fallback_doc_id] if fallback_doc_id else [])
    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            continue
        doc_id = value.strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        normalized.append(doc_id)
    return normalized


def _primary_doc_id(doc_ids: list[str]) -> str | None:
    return doc_ids[0] if doc_ids else None


def _doc_ids_to_json(doc_ids: list[str]) -> str:
    return json.dumps(_normalize_doc_ids(doc_ids), ensure_ascii=False)


def _doc_ids_from_json(value: str | None, *, fallback_doc_id: str | None = None) -> list[str]:
    if value:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            return _normalize_doc_ids([item for item in payload if isinstance(item, str)])
    return _normalize_doc_ids(fallback_doc_id=fallback_doc_id)


def _json_list(value: str) -> list[dict[str, Any]]:
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
    doc_id: str | None
    doc_ids: list[str]
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
    """SQLite-backed chat history store for the simplified single chat box."""

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
                    doc_id TEXT,
                    doc_ids_json TEXT,
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
                    tool_results_json TEXT NOT NULL,
                    usage_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )
        self._ensure_doc_ids_column()
        self._ensure_turn_metadata_columns()
        self._backfill_doc_ids_json()

    def _ensure_doc_ids_column(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("PRAGMA table_info(sessions)").fetchall()
            columns = {row["name"] for row in rows}
            if "doc_ids_json" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN doc_ids_json TEXT")

    def _ensure_turn_metadata_columns(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("PRAGMA table_info(session_turns)").fetchall()
            columns = {row["name"] for row in rows}
            if "reasoning_content" not in columns:
                connection.execute(
                    "ALTER TABLE session_turns ADD COLUMN reasoning_content TEXT NOT NULL DEFAULT ''"
                )
            if "usage_json" not in columns:
                connection.execute("ALTER TABLE session_turns ADD COLUMN usage_json TEXT")

    def _backfill_doc_ids_json(self) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, doc_id, doc_ids_json
                FROM sessions
                WHERE doc_ids_json IS NULL
                """
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE sessions SET doc_ids_json = ? WHERE id = ?",
                    (_doc_ids_to_json(_normalize_doc_ids(fallback_doc_id=row["doc_id"])), row["id"]),
                )

    def ensure_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> SessionSummary:
        resolved_doc_ids = _normalize_doc_ids(doc_ids, fallback_doc_id=doc_id)
        existing = self.get_session(session_id)
        if existing is not None:
            if doc_ids is not None and existing.doc_ids != resolved_doc_ids:
                return self.update_session(session_id, doc_ids=resolved_doc_ids)
            if doc_ids is None and doc_id is not None and existing.doc_id != doc_id:
                return self.update_session(session_id, doc_ids=resolved_doc_ids)
            return existing

        now = _now()
        resolved_title = (title or "新对话").strip() or "新对话"
        resolved_doc_id = _primary_doc_id(resolved_doc_ids)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, title, doc_id, doc_ids_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, resolved_title, resolved_doc_id, _doc_ids_to_json(resolved_doc_ids), now, now),
            )
        return SessionSummary(
            id=session_id,
            title=resolved_title,
            doc_id=resolved_doc_id,
            doc_ids=resolved_doc_ids,
            created_at=now,
            updated_at=now,
        )

    def create_session(
        self,
        session_id: str,
        *,
        title: str = "新对话",
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> SessionSummary:
        return self.ensure_session(session_id, title=title, doc_id=doc_id, doc_ids=doc_ids)

    def list_sessions(self) -> list[SessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, doc_id, doc_ids_json, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def get_session(self, session_id: str) -> SessionSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, doc_id, doc_ids_json, created_at, updated_at
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return self._row_to_summary(row) if row else None

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        doc_id: str | None | object = _UNSET,
        doc_ids: list[str] | None | object = _UNSET,
    ) -> SessionSummary:
        session = self.get_session(session_id)
        if session is None:
            fallback_doc_id = None if doc_id is _UNSET else doc_id
            requested_doc_ids = None if doc_ids is _UNSET else doc_ids
            return self.ensure_session(
                session_id,
                title=title,
                doc_id=fallback_doc_id if isinstance(fallback_doc_id, str) else None,
                doc_ids=requested_doc_ids if isinstance(requested_doc_ids, list) else None,
            )

        resolved_title = session.title if title is None else title.strip() or session.title
        if doc_ids is not _UNSET:
            resolved_doc_ids = _normalize_doc_ids(doc_ids if isinstance(doc_ids, list) else [])
        elif doc_id is not _UNSET:
            resolved_doc_ids = _normalize_doc_ids(fallback_doc_id=doc_id if isinstance(doc_id, str) else None)
        else:
            resolved_doc_ids = session.doc_ids
        resolved_doc_id = _primary_doc_id(resolved_doc_ids)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET title = ?, doc_id = ?, doc_ids_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (resolved_title, resolved_doc_id, _doc_ids_to_json(resolved_doc_ids), now, session_id),
            )
        return SessionSummary(
            id=session.id,
            title=resolved_title,
            doc_id=resolved_doc_id,
            doc_ids=resolved_doc_ids,
            created_at=session.created_at,
            updated_at=now,
        )

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def clear_document_references(self, doc_id: str) -> int:
        resolved_doc_id = doc_id.strip()
        if not resolved_doc_id:
            return 0
        now = _now()
        changed = 0
        with self._connect() as connection:
            rows = connection.execute("SELECT id, doc_id, doc_ids_json FROM sessions").fetchall()
            for row in rows:
                current_doc_ids = _doc_ids_from_json(row["doc_ids_json"], fallback_doc_id=row["doc_id"])
                next_doc_ids = [item for item in current_doc_ids if item != resolved_doc_id]
                if next_doc_ids == current_doc_ids:
                    continue
                connection.execute(
                    """
                    UPDATE sessions
                    SET doc_id = ?, doc_ids_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (_primary_doc_id(next_doc_ids), _doc_ids_to_json(next_doc_ids), now, row["id"]),
                )
                changed += 1
        return changed

    def record_turn(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        citations: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        reasoning_content: str = "",
        usage: dict[str, Any] | None = None,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> SessionTurn:
        selection_provided = doc_ids is not None or doc_id is not None
        resolved_doc_ids = _normalize_doc_ids(doc_ids, fallback_doc_id=doc_id)
        session = self.ensure_session(
            session_id,
            title=user_content[:18] or "新的财报对话",
            doc_ids=resolved_doc_ids if selection_provided else None,
        )
        if session.title == "新对话" and user_content.strip():
            self.update_session(session_id, title=user_content[:18])
        elif selection_provided:
            self.update_session(session_id, doc_ids=resolved_doc_ids)

        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO session_turns (
                    session_id,
                    user_content,
                    assistant_content,
                    reasoning_content,
                    citations_json,
                    tool_results_json,
                    usage_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_content,
                    assistant_content,
                    reasoning_content,
                    json.dumps(citations, ensure_ascii=False),
                    json.dumps(tool_results, ensure_ascii=False),
                    json.dumps(usage, ensure_ascii=False) if usage is not None else None,
                    now,
                ),
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
            tool_results=tool_results,
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
        doc_ids = _doc_ids_from_json(row["doc_ids_json"], fallback_doc_id=row["doc_id"])
        return SessionSummary(
            id=row["id"],
            title=row["title"],
            doc_id=_primary_doc_id(doc_ids),
            doc_ids=doc_ids,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
