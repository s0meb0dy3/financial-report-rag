import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.domain import ConversationState
from app.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


DEFAULT_SESSION_DB_PATH = "data/sessions.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionSummary:
    id: str
    title: str
    doc_id: str | None
    created_at: str
    updated_at: str


@dataclass
class SessionTurn:
    id: int
    session_id: str
    user_content: str
    assistant_content: str
    citations: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    created_at: str


class SessionStore(Protocol):
    def load(self, session_id: str) -> ConversationState:
        ...

    def save(self, session_id: str, state: ConversationState) -> None:
        ...


class InMemorySessionStore:
    def __init__(self):
        self._states: dict[str, ConversationState] = {}

    def load(self, session_id: str) -> ConversationState:
        state = self._states.get(session_id)
        if state is None:
            return ConversationState()
        return deepcopy(state)

    def save(self, session_id: str, state: ConversationState) -> None:
        self._states[session_id] = deepcopy(state)


def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
    }
    if isinstance(message, AssistantMessage):
        payload["tool_calls"] = [
            {
                "tool_name": tool_call.tool_name,
                "arguments": tool_call.arguments,
                "tool_call_id": tool_call.tool_call_id,
            }
            for tool_call in message.tool_calls
        ]
    if isinstance(message, ToolResultMessage):
        payload["tool_name"] = message.tool_name
        payload["tool_call_id"] = message.tool_call_id
        payload["output"] = message.output
    return payload


def _message_from_dict(payload: dict[str, Any]) -> BaseMessage:
    role = payload.get("role")
    common = {
        "content": payload.get("content", ""),
        "created_at": payload.get("created_at", _now()),
    }
    if role == "system":
        return SystemMessage(**common)
    if role == "user":
        return UserMessage(**common)
    if role == "assistant":
        return AssistantMessage(
            **common,
            tool_calls=[
                ToolCall(
                    tool_name=item.get("tool_name", ""),
                    arguments=item.get("arguments", {}),
                    tool_call_id=item.get("tool_call_id", ""),
                )
                for item in payload.get("tool_calls", [])
                if isinstance(item, dict)
            ],
        )
    if role == "tool":
        return ToolResultMessage(
            **common,
            tool_name=payload.get("tool_name", ""),
            tool_call_id=payload.get("tool_call_id", ""),
            output=payload.get("output", {}),
        )
    raise ValueError(f"Unsupported persisted message role: {role!r}")


def _state_to_json(state: ConversationState) -> str:
    return json.dumps(
        {"messages": [_message_to_dict(message) for message in state.messages]},
        ensure_ascii=False,
    )


def _state_from_json(content: str) -> ConversationState:
    payload = json.loads(content)
    messages = [
        _message_from_dict(item)
        for item in payload.get("messages", [])
        if isinstance(item, dict)
    ]
    return ConversationState(messages=messages)


class SQLiteSessionStore:
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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_states (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_content TEXT NOT NULL,
                    assistant_content TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    tool_results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )

    def ensure_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        doc_id: str | None = None,
    ) -> SessionSummary:
        existing = self.get_session(session_id)
        if existing is not None:
            if doc_id is not None and existing.doc_id != doc_id:
                return self.update_session(session_id, doc_id=doc_id)
            return existing

        now = _now()
        resolved_title = (title or "新的财报对话").strip() or "新的财报对话"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, title, doc_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, resolved_title, doc_id, now, now),
            )
        return SessionSummary(
            id=session_id,
            title=resolved_title,
            doc_id=doc_id,
            created_at=now,
            updated_at=now,
        )

    def create_session(
        self,
        session_id: str,
        *,
        title: str = "新的财报对话",
        doc_id: str | None = None,
    ) -> SessionSummary:
        return self.ensure_session(session_id, title=title, doc_id=doc_id)

    def list_sessions(self) -> list[SessionSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, doc_id, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def get_session(self, session_id: str) -> SessionSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, title, doc_id, created_at, updated_at
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
        doc_id: str | None = None,
    ) -> SessionSummary:
        session = self.get_session(session_id)
        if session is None:
            return self.ensure_session(session_id, title=title, doc_id=doc_id)

        resolved_title = session.title if title is None else title.strip() or session.title
        resolved_doc_id = session.doc_id if doc_id is None else doc_id
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET title = ?, doc_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (resolved_title, resolved_doc_id, now, session_id),
            )
        return SessionSummary(
            id=session.id,
            title=resolved_title,
            doc_id=resolved_doc_id,
            created_at=session.created_at,
            updated_at=now,
        )

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM session_states WHERE session_id = ?", (session_id,))
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def record_turn(
        self,
        session_id: str,
        *,
        user_content: str,
        assistant_content: str,
        citations: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        doc_id: str | None = None,
    ) -> SessionTurn:
        session = self.ensure_session(
            session_id,
            title=user_content[:18] or "新的财报对话",
            doc_id=doc_id,
        )
        title = session.title
        if title == "新的财报对话" and user_content.strip():
            self.update_session(session_id, title=user_content[:18], doc_id=doc_id)
        elif doc_id is not None:
            self.update_session(session_id, doc_id=doc_id)

        now = _now()
        citations_json = json.dumps(citations, ensure_ascii=False)
        tool_results_json = json.dumps(tool_results, ensure_ascii=False)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO session_turns (
                    session_id,
                    user_content,
                    assistant_content,
                    citations_json,
                    tool_results_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_content,
                    assistant_content,
                    citations_json,
                    tool_results_json,
                    now,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            turn_id = int(cursor.lastrowid)
        return SessionTurn(
            id=turn_id,
            session_id=session_id,
            user_content=user_content,
            assistant_content=assistant_content,
            citations=citations,
            tool_results=tool_results,
            created_at=now,
        )

    def list_turns(self, session_id: str) -> list[SessionTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, user_content, assistant_content,
                       citations_json, tool_results_json, created_at
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
                citations=json.loads(row["citations_json"]),
                tool_results=json.loads(row["tool_results_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def load(self, session_id: str) -> ConversationState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM session_states WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return ConversationState()
        return _state_from_json(row["state_json"])

    def save(self, session_id: str, state: ConversationState) -> None:
        self.ensure_session(session_id)
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_states (session_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, _state_to_json(state), now),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> SessionSummary:
        return SessionSummary(
            id=row["id"],
            title=row["title"],
            doc_id=row["doc_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
