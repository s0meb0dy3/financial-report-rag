import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.chat_service import ChatService, NO_EVIDENCE_ANSWER
from app.rag import RagCitation, RagEvidence, RagResult
from app.session import SQLiteSessionStore


def make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def make_stream(*parts: str):
    for part in parts:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=part))])


def make_reasoning_stream():
    yield SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="先找证据。", content=None))],
    )
    yield SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="营业收入"))],
    )
    yield SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        ),
        choices=[],
    )
    yield SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="为 100 亿元。"))],
    )


def rag_result(*, with_context: bool = True) -> RagResult:
    evidences = (
        [
            RagEvidence(
                doc_id="doc-a",
                doc_name="doc-a.pdf",
                page=8,
                text="营业收入为 100 亿元。",
                score=0.9,
            )
        ]
        if with_context
        else []
    )
    citations = [RagCitation(doc_id="doc-a", doc_name="doc-a.pdf", page=8)] if with_context else []
    return RagResult(
        query="营业收入是多少？",
        retrieval_queries=["营业收入是多少？"],
        evidences=evidences,
        tables=[],
        citations=citations,
        metadata={},
    )


class FakeRagService:
    def __init__(self, result: RagResult):
        self.result = result
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.result

    def close(self):
        pass


class ChatServiceTests(unittest.TestCase):
    def test_ask_generates_answer_from_rag_and_records_turn(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response("营业收入为 100 亿元。")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                rag_service=FakeRagService(rag_result()),
                session_store=store,
                client=client,
                model="test-model",
            )

            result = service.ask(" 营业收入是多少？ ", session_id="session-1", top_k=4)
            turns = store.list_turns("session-1")

        self.assertEqual(result.answer, "营业收入为 100 亿元。")
        self.assertEqual(result.citations[0].page, 8)
        self.assertEqual(turns[0].assistant_content, "营业收入为 100 亿元。")
        self.assertIn("检索证据", client.chat.completions.create.call_args.kwargs["messages"][-1]["content"])

    def test_ask_returns_no_evidence_answer_without_llm_call(self) -> None:
        client = MagicMock()

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                rag_service=FakeRagService(rag_result(with_context=False)),
                session_store=store,
                client=client,
                model="test-model",
            )

            result = service.ask("没有证据的问题", session_id="session-1")

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        client.chat.completions.create.assert_not_called()

    def test_stream_emits_minimal_events_and_records_turn(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_stream("营业收入", "为 100 亿元。")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                rag_service=FakeRagService(rag_result()),
                session_store=store,
                client=client,
                model="test-model",
            )

            events = list(service.stream("营业收入是多少？", session_id="session-1"))
            turns = store.list_turns("session-1")

        self.assertEqual(
            [event["event"] for event in events],
            ["session", "tool", "tool", "status", "usage", "answer_delta", "answer_delta", "final"],
        )
        self.assertEqual(events[-1]["data"]["answer"], "营业收入为 100 亿元。")
        self.assertEqual(turns[0].assistant_content, "营业收入为 100 亿元。")

    def test_stream_emits_reasoning_and_usage(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_reasoning_stream()

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                rag_service=FakeRagService(rag_result()),
                session_store=store,
                client=client,
                model="test-model",
                context_window_tokens=1000,
            )

            events = list(service.stream("营业收入是多少？", session_id="session-1"))

        reasoning = [event for event in events if event["event"] == "reasoning_delta"]
        usage = [event for event in events if event["event"] == "usage"][-1]["data"]
        self.assertEqual(reasoning[0]["data"]["content"], "先找证据。")
        self.assertEqual(events[-1]["data"]["reasoning_content"], "先找证据。")
        self.assertEqual(usage["prompt_tokens"], 120)
        self.assertEqual(usage["reasoning_tokens"], 12)
        self.assertEqual(usage["cached_tokens"], 40)
        self.assertEqual(usage["context_window_tokens"], 1000)
        self.assertFalse(usage["estimated"])

    def test_mimo_reasoning_history_is_passed_back_when_enabled(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response("第二轮回答")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn(
                "session-1",
                user_content="第一问",
                assistant_content="第一轮回答",
                reasoning_content="第一轮思考",
                citations=[],
                tool_results=[],
            )
            service = ChatService(
                rag_service=FakeRagService(rag_result()),
                session_store=store,
                client=client,
                model="mimo-v2.5-pro",
                thinking_enabled=True,
                pass_reasoning_history=True,
            )

            service.ask("第二问", session_id="session-1")

        kwargs = client.chat.completions.create.call_args.kwargs
        assistant_history = kwargs["messages"][2]
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        self.assertEqual(assistant_history["role"], "assistant")
        self.assertEqual(assistant_history["reasoning_content"], "第一轮思考")


if __name__ == "__main__":
    unittest.main()
