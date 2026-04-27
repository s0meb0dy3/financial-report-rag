import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.agent import AgentLoop
from app.api import create_app
from app.domain import DocumentRef
from app.session import SQLiteSessionStore


def make_message(*, content: str | None = None, tool_calls: list | None = None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    return message


def make_response(message) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


class FakeStreamingLoop:
    def __init__(self) -> None:
        self.retriever = MagicMock()

    def run_turn_stream(self, question, session_id=None, top_k=None, doc_id=None, doc_ids=None):
        yield {"event": "status", "data": {"message": "检索证据"}}
        yield {
            "event": "tool_result",
            "data": {
                "tool_name": "search_reports",
                "arguments": {"query": question, "top_k": top_k},
                "output": {"results": []},
                "tool_call_id": "call-1",
            },
        }
        yield {"event": "answer_delta", "data": {"content": "营业总收入"}}
        yield {"event": "answer_delta", "data": {"content": "为 100 亿元。"}}
        yield {
            "event": "final",
            "data": {
                "answer": "营业总收入为 100 亿元。",
                "citations": [{"doc_id": doc_id or "doc-a", "doc_name": "doc-a.pdf", "page": 12}],
                "tool_results": [
                    {
                        "tool_name": "search_reports",
                        "arguments": {"query": question, "top_k": top_k},
                        "output": {"results": []},
                        "tool_call_id": "call-1",
                    }
                ],
            },
        }


class ApiTests(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        loop = MagicMock()

        with TestClient(create_app(agent_loop=loop)) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        loop.close.assert_not_called()

    def test_documents_returns_indexed_reports(self) -> None:
        loop = MagicMock()
        loop.retriever.list_documents.return_value = [
            DocumentRef(doc_id="moutai", doc_name="贵州茅台2024年报.pdf")
        ]

        with TestClient(create_app(agent_loop=loop)) as client:
            response = client.get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"documents": [{"doc_id": "moutai", "doc_name": "贵州茅台2024年报.pdf"}]},
        )

    def test_chat_runs_agent_turn_with_request_options(self) -> None:
        loop = MagicMock()
        loop.run_turn.return_value = {
            "answer": "营业总收入为 1741.44 亿元。",
            "citations": [{"doc_id": "moutai", "doc_name": "贵州茅台2024年报.pdf", "page": 12}],
            "tool_results": [
                {
                    "tool_name": "search_reports",
                    "arguments": {"query": "营业总收入", "top_k": 5},
                    "output": {"results": []},
                    "tool_call_id": "call-1",
                }
            ],
        }

        with TestClient(create_app(agent_loop=loop)) as client:
            response = client.post(
                "/chat",
                json={
                    "question": " 营业总收入是多少？ ",
                    "session_id": "web-1",
                    "top_k": 5,
                    "doc_id": "moutai",
                    "include_tool_results": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "营业总收入为 1741.44 亿元。")
        self.assertEqual(response.json()["citations"][0]["page"], 12)
        self.assertEqual(response.json()["tool_results"][0]["tool_call_id"], "call-1")
        loop.run_turn.assert_called_once_with(
            "营业总收入是多少？",
            session_id="web-1",
            top_k=5,
            doc_id="moutai",
            doc_ids=None,
        )

    def test_chat_accepts_multiple_document_filters(self) -> None:
        loop = MagicMock()
        loop.run_turn.return_value = {
            "answer": "对比回答",
            "citations": [],
            "tool_results": [],
        }

        with TestClient(create_app(agent_loop=loop)) as client:
            response = client.post(
                "/chat",
                json={
                    "question": "对比美的和长江电力的营收",
                    "session_id": "web-1",
                    "doc_ids": ["midea", "cyc"],
                },
            )

        self.assertEqual(response.status_code, 200)
        loop.run_turn.assert_called_once_with(
            "对比美的和长江电力的营收",
            session_id="web-1",
            top_k=None,
            doc_id=None,
            doc_ids=["midea", "cyc"],
        )

    def test_chat_hides_tool_results_by_default(self) -> None:
        loop = MagicMock()
        loop.run_turn.return_value = {
            "answer": "回答",
            "citations": [],
            "tool_results": [{"tool_name": "search_reports", "arguments": {}, "output": {}}],
        }

        with TestClient(create_app(agent_loop=loop)) as client:
            response = client.post("/chat", json={"question": "问题"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tool_results"], [])

    def test_session_api_creates_lists_updates_details_and_deletes(self) -> None:
        loop = MagicMock()

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            with TestClient(create_app(agent_loop=loop, session_store=store)) as client:
                created = client.post(
                    "/sessions",
                    json={"title": "测试会话", "doc_ids": ["moutai", "pingan"]},
                )
                session_id = created.json()["id"]

                listed = client.get("/sessions")
                updated = client.patch(
                    f"/sessions/{session_id}",
                    json={"title": "更新后的会话", "doc_ids": ["pingan"]},
                )
                store.record_turn(
                    session_id,
                    user_content="问题",
                    assistant_content="回答",
                    citations=[{"doc_id": "pingan", "doc_name": "doc.pdf", "page": 3}],
                    tool_results=[],
                    doc_id="pingan",
                )
                detail = client.get(f"/sessions/{session_id}")
                deleted = client.delete(f"/sessions/{session_id}")
                missing = client.get(f"/sessions/{session_id}")

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["doc_ids"], ["moutai", "pingan"])
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["sessions"][0]["title"], "测试会话")
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["doc_id"], "pingan")
        self.assertEqual(updated.json()["doc_ids"], ["pingan"])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(len(detail.json()["messages"]), 2)
        self.assertEqual(detail.json()["messages"][1]["citations"][0]["page"], 3)
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(missing.status_code, 404)

    def test_chat_records_turn_in_sqlite_store(self) -> None:
        client_mock = MagicMock()
        client_mock.chat.completions.create.return_value = make_response(make_message(content="持久化回答"))

        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = []
        retriever = MagicMock()

        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)
            loop = AgentLoop(
                api_key="test-key",
                client=client_mock,
                retriever=retriever,
                tool_registry=tool_registry,
                session_store=store,
            )
            with TestClient(create_app(agent_loop=loop, session_store=store)) as client:
                response = client.post(
                    "/chat",
                    json={
                        "question": "问题",
                        "session_id": "session-1",
                        "doc_id": "moutai",
                    },
                )

            turns = SQLiteSessionStore(db_path).list_turns("session-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "持久化回答")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].user_content, "问题")
        self.assertEqual(turns[0].assistant_content, "持久化回答")

    def test_chat_stream_returns_sse_events_in_order(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            with TestClient(
                create_app(agent_loop=FakeStreamingLoop(), session_store=store)
            ) as client:
                response = client.post(
                    "/chat/stream",
                    json={
                        "question": "营业总收入是多少？",
                        "session_id": "session-1",
                        "top_k": 5,
                        "doc_id": "moutai",
                    },
                )

        body = response.text
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        status_index = body.index("event: status")
        tool_index = body.index("event: tool_result")
        delta_index = body.index("event: answer_delta")
        final_index = body.index("event: final")
        self.assertLess(status_index, tool_index)
        self.assertLess(tool_index, delta_index)
        self.assertLess(delta_index, final_index)
        self.assertIn('"answer": "营业总收入为 100 亿元。"', body)


if __name__ == "__main__":
    unittest.main()
