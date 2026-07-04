import unittest
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import create_app
from app.chat_service import ChatService
from app.documents import DocumentService
from app.session import SQLiteSessionStore


def make_pdf_bytes(text: str = "") -> bytes:
    import pymupdf

    pdf = pymupdf.open()
    try:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        return pdf.tobytes()
    finally:
        pdf.close()


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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_document_service(root: Path) -> DocumentService:
    pdf = root / "raw" / "report.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4")
    artifact = root / "mineru" / "doc-a"
    write_json(
        artifact / "manifest.json",
        {"doc_id": "doc-a", "file_name": "report.pdf", "source_path": str(pdf)},
    )
    write_json(
        artifact / "content_list_v2.json",
        [
            [
                {
                    "type": "paragraph",
                    "content": {"paragraph_content": [{"type": "text", "content": "第一页内容"}]},
                }
            ]
        ],
    )
    return DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")


class ApiTests(unittest.TestCase):
    def test_health_returns_ok(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_runtime_config_exposes_safe_model_status(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CHAT_API_KEY": "secret",
                "CHAT_BASE_URL": "https://example.test/v1",
                "CHAT_MODEL": "custom-model",
            },
            clear=True,
        ):
            with TemporaryDirectory() as directory:
                store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
                with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                    response = client.get("/runtime/config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chat_model"], "custom-model")
        self.assertEqual(response.json()["chat_base_url"], "https://example.test/v1")
        self.assertTrue(response.json()["api_key_configured"])
        self.assertFalse(response.json()["mineru_api_key_configured"])
        self.assertNotIn("secret", response.text)

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

    def test_chat_stream_hides_internal_error_details(self) -> None:
        client_mock = MagicMock()
        client_mock.chat.completions.create.side_effect = RuntimeError("secret-token")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(session_store=store, client=client_mock, model="test-model")
            with TestClient(create_app(chat_service=service, session_store=store)) as client:
                with self.assertLogs("app.api", level="ERROR") as logs:
                    response = client.post(
                        "/chat/stream",
                        json={"question": "营业总收入是多少？", "session_id": "session-1"},
                    )

        body = response.text
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", body)
        self.assertIn("请求失败，请查看后端日志。", body)
        self.assertNotIn("secret-token", body)
        self.assertIn("RuntimeError", logs.output[0])
        self.assertNotIn("secret-token", logs.output[0])

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

    def test_document_endpoints_list_read_and_serve_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteSessionStore(root / "sessions.sqlite3")
            doc_service = build_document_service(root)
            with TestClient(
                create_app(
                    chat_service=build_test_service(store),
                    session_store=store,
                    document_service=doc_service,
                )
            ) as client:
                docs = client.get("/documents")
                page = client.get("/documents/doc-a/pages/1")
                pdf = client.get("/documents/doc-a/pdf")

        self.assertEqual(docs.status_code, 200)
        self.assertEqual(docs.json()[0]["id"], "doc-a")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.json()["text"], "第一页内容")
        self.assertEqual(pdf.status_code, 200)
        self.assertIn("application/pdf", pdf.headers["content-type"])

    def test_session_rename_and_delete_endpoints(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn("session-1", user_content="问题", assistant_content="回答", citations=[])
            with TestClient(create_app(chat_service=build_test_service(store), session_store=store)) as client:
                renamed = client.patch("/sessions/session-1", json={"title": "新标题"})
                deleted = client.delete("/sessions/session-1")
                missing = client.get("/sessions/session-1")

        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "新标题")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(missing.status_code, 404)

    def test_document_upload_and_delete_endpoints(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteSessionStore(root / "sessions.sqlite3")
            doc_service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")
            with TestClient(
                create_app(
                    chat_service=build_test_service(store),
                    session_store=store,
                    document_service=doc_service,
                )
            ) as client:
                uploaded = client.post(
                    "/documents?filename=upload.pdf",
                    content=make_pdf_bytes("uploaded page text"),
                    headers={"Content-Type": "application/pdf"},
                )
                doc_id = uploaded.json()["id"]
                listed = client.get("/documents")
                page = client.get(f"/documents/{doc_id}/pages/1")
                deleted = client.delete(f"/documents/{doc_id}")

        self.assertEqual(uploaded.status_code, 200)
        self.assertEqual(uploaded.json()["parsed"], True)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["name"], "upload.pdf")
        self.assertEqual(page.status_code, 200)
        self.assertIn("uploaded page text", page.json()["text"])
        self.assertEqual(deleted.status_code, 204)

    def test_document_upload_rejects_invalid_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteSessionStore(root / "sessions.sqlite3")
            doc_service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")
            with TestClient(
                create_app(
                    chat_service=build_test_service(store),
                    session_store=store,
                    document_service=doc_service,
                )
            ) as client:
                response = client.post(
                    "/documents?filename=bad.pdf",
                    content=b"not a pdf",
                    headers={"Content-Type": "application/pdf"},
                )

        self.assertEqual(response.status_code, 422)
        self.assertIn("invalid", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
