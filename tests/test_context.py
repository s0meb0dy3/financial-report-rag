import unittest

from app.context import ContextBuilder
from app.domain import ConversationState
from app.messages import AssistantMessage, SystemMessage, ToolResultMessage, UserMessage


class ContextBuilderTests(unittest.TestCase):
    def test_build_includes_system_history_and_new_user_message(self) -> None:
        state = ConversationState(
            messages=[
                SystemMessage(content="system"),
                UserMessage(content="上一轮问题"),
                AssistantMessage(content="上一轮回答"),
            ]
        )
        builder = ContextBuilder()

        messages = builder.build(state, "这轮问题")

        self.assertEqual([message.role for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1].content, "这轮问题")

    def test_build_preserves_tool_result_messages_for_evidence_injection(self) -> None:
        state = ConversationState(
            messages=[
                SystemMessage(content="system"),
                ToolResultMessage(
                    tool_name="search_reports",
                    tool_call_id="call-1",
                    output={"query": "营收", "results": [{"doc_name": "doc-a.pdf"}]},
                ),
            ]
        )
        builder = ContextBuilder()

        messages = builder.build(state, "总结一下")

        self.assertEqual(messages[1].role, "tool")
        self.assertEqual(messages[1].tool_name, "search_reports")


if __name__ == "__main__":
    unittest.main()
