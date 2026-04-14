import unittest

from app.domain import ConversationState
from app.messages import SystemMessage
from app.session import InMemorySessionStore


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


if __name__ == "__main__":
    unittest.main()
