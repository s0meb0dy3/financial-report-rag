import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api import create_app
from app.chat_service import ChatService
from app.domain import Evidence
from app.rag import RagService
from app.session import SQLiteSessionStore


def make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def make_stream(*parts: str):
    for part in parts:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=part))])


class FakeRetriever:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    def search(self, query, top_k=3, filters=None):
        self.search_calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [
            Evidence(
                doc_id="moutai",
                doc_name="贵州茅台2024年报.pdf",
                page=12,
                text="营业总收入为 1741.44 亿元。",
                score=0.91,
                chunk_id="chunk-1",
            )
        ]

    def get_last_retrieval_queries(self):
        return ["营业总收入是多少？", "主要会计数据 营业总收入"]

    def list_documents(self):
        return []


class FakeTableRepository:
    def search_tables(self, **kwargs):
        return [
            {
                "table_id": "table-1",
                "doc_id": "moutai",
                "doc_name": "贵州茅台2024年报.pdf",
                "title": "主要会计数据",
                "page_start": 12,
                "page_end": 12,
                "preview_matrix": [["指标", "2024"], ["营业总收入", "1741.44"]],
                "score": 3.0,
            }
        ]


def build_test_service(store: SQLiteSessionStore) -> ChatService:
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kwargs: (
        make_stream("营业总收入", "为 1741.44 亿元。")
        if kwargs.get("stream")
        else make_response("营业总收入为 1741.44 亿元。")
    )
    rag_service = RagService(
        retriever=FakeRetriever(),
        table_repository=FakeTableRepository(),
    )
    return ChatService(
        rag_service=rag_service,
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

    def test_rag_retrieve_returns_evidence_and_tables(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = build_test_service(store)
            with TestClient(create_app(chat_service=service, session_store=store)) as client:
                response = client.post(
                    "/rag/retrieve",
                    json={"query": "营业总收入是多少？", "top_k": 4, "doc_ids": ["moutai"]},
                )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["retrieval_queries"][1], "主要会计数据 营业总收入")
        self.assertEqual(payload["evidences"][0]["page"], 12)
        self.assertEqual(payload["tables"][0]["table_id"], "table-1")
        self.assertEqual(payload["citations"][0]["doc_id"], "moutai")

    def test_chat_returns_answer_session_and_citations(self) -> None:
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
        self.assertEqual(payload["answer"], "营业总收入为 1741.44 亿元。")
        self.assertEqual(payload["citations"][0]["page"], 12)
        self.assertEqual(turns[0].assistant_content, "营业总收入为 1741.44 亿元。")

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
        self.assertNotIn("event: tool_result", body)
        self.assertIn('"answer": "营业总收入为 1741.44 亿元。"', body)

    def test_get_session_restores_chat_history(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn(
                "session-1",
                user_content="问题",
                assistant_content="回答",
                reasoning_content="思考",
                citations=[{"doc_id": "moutai", "doc_name": "doc.pdf", "page": 3}],
                tool_results=[{"id": "retrieve_context", "name": "retrieve_context", "status": "done"}],
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
        self.assertEqual(payload["messages"][1]["tool_results"][0]["id"], "retrieve_context")
        self.assertEqual(payload["messages"][1]["usage"]["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
