import json
import unittest

from app.messages import (
    AssistantMessage,
    OpenAIMessageAdapter,
    SystemMessage,
    ToolCall,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)


class MessageSchemaTests(unittest.TestCase):
    def test_to_openai_messages_serializes_structured_messages(self) -> None:
        messages = [
            SystemMessage(content="system"),
            UserMessage(content="question"),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="search_reports",
                        arguments={"query": "营收"},
                        tool_call_id="call-1",
                    )
                ],
            ),
            ToolResultMessage(
                tool_name="search_reports",
                tool_call_id="call-1",
                output={"query": "营收", "results": [{"doc_name": "doc-a.pdf"}]},
            ),
        ]

        payload = OpenAIMessageAdapter.to_openai(messages)

        self.assertEqual(payload[0], {"role": "system", "content": "system"})
        self.assertEqual(payload[1], {"role": "user", "content": "question"})
        self.assertEqual(payload[2]["role"], "assistant")
        self.assertEqual(payload[2]["tool_calls"][0]["function"]["name"], "search_reports")
        self.assertEqual(payload[3]["role"], "tool")
        self.assertEqual(payload[3]["tool_call_id"], "call-1")
        self.assertIn("doc-a.pdf", payload[3]["content"])

    def test_from_openai_response_builds_assistant_message_with_tool_calls(self) -> None:
        tool_call_payload = {
            "id": "call-1",
            "function": {
                "name": "search_reports",
                "arguments": json.dumps({"query": "营收"}, ensure_ascii=False),
            },
        }

        message = OpenAIMessageAdapter.from_openai_response(
            content="",
            tool_calls=[tool_call_payload],
        )

        self.assertIsInstance(message, AssistantMessage)
        self.assertEqual(message.tool_calls[0].tool_name, "search_reports")
        self.assertEqual(message.tool_calls[0].arguments, {"query": "营收"})

    def test_tool_call_message_keeps_structured_arguments(self) -> None:
        message = ToolCallMessage(
            tool_name="search_reports",
            tool_call_id="call-1",
            arguments={"query": "营收", "top_k": 3},
        )

        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.arguments["top_k"], 3)


if __name__ == "__main__":
    unittest.main()
