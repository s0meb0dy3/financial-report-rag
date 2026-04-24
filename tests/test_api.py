import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api import create_app
from app.domain import DocumentRef


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


if __name__ == "__main__":
    unittest.main()
