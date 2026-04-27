import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.domain import ConversationState
from app.messages import AssistantMessage, SystemMessage, ToolCall, ToolResultMessage, UserMessage
from app.session import InMemorySessionStore, SQLiteSessionStore


class SessionStoreTests(unittest.TestCase):
    def test_conversation_state_rejects_unstructured_dict_messages(self) -> None:
        with self.assertRaises(TypeError):
            ConversationState(messages=[{"role": "user", "content": "hi"}])

    def test_load_returns_saved_state(self) -> None:
        store = InMemorySessionStore()
        state = ConversationState(messages=[SystemMessage(content="system")])

        store.save("session-1", state)
        loaded = store.load("session-1")

        self.assertEqual(loaded.messages[0].content, "system")

    def test_load_returns_empty_state_for_missing_session(self) -> None:
        store = InMemorySessionStore()

        loaded = store.load("missing")

        self.assertEqual(loaded.messages, [])

    def test_sqlite_store_persists_structured_conversation_state(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)
            state = ConversationState(
                messages=[
                    UserMessage(content="营业总收入是多少？"),
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            ToolCall(
                                tool_name="search_reports",
                                arguments={"query": "营业总收入"},
                                tool_call_id="call-1",
                            )
                        ],
                    ),
                    ToolResultMessage(
                        tool_name="search_reports",
                        tool_call_id="call-1",
                        output={"results": [{"doc_name": "doc-a.pdf"}]},
                    ),
                    AssistantMessage(content="营业总收入为 100 亿元。"),
                ]
            )

            store.save("session-1", state)
            loaded = SQLiteSessionStore(db_path).load("session-1")

            self.assertIsInstance(loaded.messages[0], UserMessage)
            self.assertIsInstance(loaded.messages[1], AssistantMessage)
            self.assertEqual(loaded.messages[1].tool_calls[0].tool_name, "search_reports")
            self.assertIsInstance(loaded.messages[2], ToolResultMessage)
            self.assertEqual(loaded.messages[2].output["results"][0]["doc_name"], "doc-a.pdf")
            self.assertEqual(loaded.messages[3].content, "营业总收入为 100 亿元。")

    def test_sqlite_store_records_turns_and_session_summary(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)

            store.record_turn(
                "session-1",
                user_content="营业总收入是多少？",
                assistant_content="营业总收入为 100 亿元。",
                citations=[{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 12}],
                tool_results=[
                    {
                        "tool_name": "search_reports",
                        "arguments": {"query": "营业总收入"},
                        "output": {"results": []},
                        "tool_call_id": "call-1",
                    }
                ],
                doc_id="doc-a",
            )

            reloaded = SQLiteSessionStore(db_path)
            session = reloaded.get_session("session-1")
            turns = reloaded.list_turns("session-1")

            self.assertIsNotNone(session)
            self.assertEqual(session.doc_id, "doc-a")
            self.assertEqual(session.doc_ids, ["doc-a"])
            self.assertEqual(session.title, "营业总收入是多少？")
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0].citations[0]["page"], 12)
            self.assertEqual(turns[0].tool_results[0]["tool_call_id"], "call-1")

    def test_sqlite_store_persists_multiple_document_selection(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)

            created = store.create_session(
                "session-1",
                title="对比会话",
                doc_ids=["doc-a", "doc-b", "doc-a"],
            )
            updated = store.update_session("session-1", doc_ids=["doc-b", "doc-c"])
            store.clear_document_references("doc-b")
            reloaded = SQLiteSessionStore(db_path).get_session("session-1")

        self.assertEqual(created.doc_id, "doc-a")
        self.assertEqual(created.doc_ids, ["doc-a", "doc-b"])
        self.assertEqual(updated.doc_id, "doc-b")
        self.assertEqual(updated.doc_ids, ["doc-b", "doc-c"])
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.doc_id, "doc-c")
        self.assertEqual(reloaded.doc_ids, ["doc-c"])


if __name__ == "__main__":
    unittest.main()
