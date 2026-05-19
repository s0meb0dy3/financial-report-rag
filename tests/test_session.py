import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.session import SQLiteSessionStore


class SessionStoreTests(unittest.TestCase):
    def test_sqlite_store_records_turns_and_session_summary(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)

            store.record_turn(
                "session-1",
                user_content="营业总收入是多少？",
                assistant_content="营业总收入为 100 亿元。",
                reasoning_content="先检索财报证据。",
                citations=[{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 12}],
                tool_results=[],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
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
        self.assertEqual(turns[0].reasoning_content, "先检索财报证据。")
        self.assertEqual(turns[0].citations[0]["page"], 12)
        self.assertEqual(turns[0].tool_results, [])
        self.assertEqual(turns[0].usage["total_tokens"], 120)

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

    def test_delete_session_removes_turns(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)
            store.record_turn(
                "session-1",
                user_content="问题",
                assistant_content="回答",
                citations=[],
                tool_results=[],
            )

            deleted = store.delete_session("session-1")
            missing = store.get_session("session-1")
            turns = store.list_turns("session-1")

        self.assertTrue(deleted)
        self.assertIsNone(missing)
        self.assertEqual(turns, [])


if __name__ == "__main__":
    unittest.main()
