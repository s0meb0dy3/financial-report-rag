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
                reasoning_content="先思考。",
                citations=[{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 12}],
                tool_results=[{"id": "call-1", "name": "tavily_search", "status": "done"}],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            )

            reloaded = SQLiteSessionStore(db_path)
            session = reloaded.get_session("session-1")
            turns = reloaded.list_turns("session-1")

        self.assertIsNotNone(session)
        self.assertEqual(session.title, "营业总收入是多少？")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].reasoning_content, "先思考。")
        self.assertEqual(turns[0].citations[0]["page"], 12)
        self.assertEqual(turns[0].tool_results[0]["name"], "tavily_search")
        self.assertEqual(turns[0].usage["total_tokens"], 120)

    def test_update_session_title(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)

            created = store.create_session("session-1", title="旧标题")
            updated = store.update_session("session-1", title="新标题")
            reloaded = SQLiteSessionStore(db_path).get_session("session-1")

        self.assertEqual(created.title, "旧标题")
        self.assertEqual(updated.title, "新标题")
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.title, "新标题")

    def test_delete_session_removes_turns(self) -> None:
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "sessions.sqlite3"
            store = SQLiteSessionStore(db_path)
            store.record_turn(
                "session-1",
                user_content="问题",
                assistant_content="回答",
                citations=[],
            )

            deleted = store.delete_session("session-1")
            missing = store.get_session("session-1")
            turns = store.list_turns("session-1")

        self.assertTrue(deleted)
        self.assertIsNone(missing)
        self.assertEqual(turns, [])

    def test_record_turn_can_require_existing_session(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")

            with self.assertRaises(ValueError):
                store.record_turn(
                    "deleted-session",
                    user_content="问题",
                    assistant_content="回答",
                    citations=[],
                    create_session=False,
                )

            self.assertIsNone(store.get_session("deleted-session"))


if __name__ == "__main__":
    unittest.main()
