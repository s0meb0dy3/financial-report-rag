import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api import create_app
from app.chat_service import ChatService
from app.session import SQLiteSessionStore


def make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def make_stream(*parts: str):
    for part in parts:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=part))])


def build_test_service(store: SQLiteSessionStore) -> ChatService:
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kwargs: (
        make_stream("测试", "回答。")
        if kwargs.get("stream")
        else make_response("测试回答。")
    )
    return ChatService(
        session_store=store,
        client=client,
        model="test-model",
    )


class ApiTests(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_chat_returns_answer_and_session(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)
            with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                response = client.post(
                    "/chat",
                    json={"question": " 营业总收入是多少？ ", "session_id": "session-1"},
                )

            turns = SQLiteSessionStore(db_path).list_turns("session-1")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["answer"], "测试回答。")
        self.assertEqual(payload["citations"], [])
        self.assertEqual(payload["tool_results"], [])
        self.assertEqual(turns[0].assistant_content, "测试回答。")

    def test_chat_stream_returns_minimal_sse_events(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                response = client.post(
                    "/chat/stream",
                    json={"question": "营业总收入是多少？", "session_id": "session-1"},
                )

        body = response.text
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("event: session", body)
        self.assertIn("event: status", body)
        self.assertIn("event: answer_delta", body)
        self.assertIn("event: final", body)
        self.assertNotIn("event: tool", body)
        self.assertIn('"answer": "测试回答。"', body)

    def test_get_session_restores_chat_history(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn(
                "session-1",
                user_content="问题",
                assistant_content="回答",
                reasoning_content="思考",
                citations=[{"doc_id": "moutai", "doc_name": "doc.pdf", "page": 3}],
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )
            with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                response = client.get("/sessions/session-1")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][1]["citations"][0]["page"], 3)
        self.assertEqual(payload["messages"][1]["reasoning_content"], "思考")
        self.assertEqual(payload["messages"][1]["tool_results"], [])
        self.assertEqual(payload["messages"][1]["usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
