import unittest
from unittest.mock import MagicMock

from app.domain import ConversationState
from app.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from app.runtime import SingleAgentRuntime
from app.session import InMemorySessionStore


class FakeStreamingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate_stream(self, messages, tool_definitions=None):
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "message",
                "message": AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            tool_name="search_reports",
                            arguments={"query": "营收"},
                            tool_call_id="call-1",
                        )
                    ],
                ),
            }
            return

        yield {"type": "content_delta", "content": "最终"}
        yield {"type": "content_delta", "content": "回答"}
        yield {"type": "message", "message": AssistantMessage(content="最终回答")}


class RuntimeTests(unittest.TestCase):
    def test_run_turn_returns_turn_result_with_updated_state(self) -> None:
        llm_client = MagicMock()
        llm_client.generate.return_value = AssistantMessage(content="直接回答")
        runtime = SingleAgentRuntime(
            llm_client=llm_client,
            tool_registry=MagicMock(),
            session_store=InMemorySessionStore(),
        )

        result = runtime.run_turn("营业总收入是多少？", session_id="s1")

        self.assertEqual(result.answer, "直接回答")
        self.assertEqual(result.updated_state.messages[-1].content, "直接回答")
        self.assertEqual(result.tool_traces, [])

    def test_run_turn_records_tool_trace(self) -> None:
        llm_client = MagicMock()
        llm_client.generate.side_effect = [
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
            AssistantMessage(content="最终回答"),
        ]

        tool_registry = MagicMock()
        tool_registry.execute.return_value = {"query": "营收", "results": [{"doc_name": "doc-a.pdf"}]}

        runtime = SingleAgentRuntime(
            llm_client=llm_client,
            tool_registry=tool_registry,
            session_store=InMemorySessionStore(),
        )

        result = runtime.run_turn("营业总收入是多少？", session_id="s1")

        self.assertEqual(result.answer, "最终回答")
        self.assertEqual(result.tool_traces[0].tool_name, "search_reports")
        self.assertEqual(result.tool_traces[0].output["results"][0]["doc_name"], "doc-a.pdf")
        self.assertIsInstance(result.updated_state.messages[0], UserMessage)
        self.assertIsInstance(result.updated_state.messages[1], AssistantMessage)
        self.assertEqual(result.updated_state.messages[1].tool_calls[0].tool_name, "search_reports")
        self.assertIsInstance(result.updated_state.messages[2], ToolResultMessage)

    def test_run_turn_stream_emits_tool_result_delta_and_final_events(self) -> None:
        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = [
            {"type": "function", "function": {"name": "search_reports"}}
        ]
        tool_registry.execute.return_value = {
            "query": "营收",
            "results": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 12}],
        }
        runtime = SingleAgentRuntime(
            llm_client=FakeStreamingLLM(),
            tool_registry=tool_registry,
            session_store=InMemorySessionStore(),
        )

        events = list(runtime.run_turn_stream("营业总收入是多少？", session_id="s1"))
        event_names = [event["event"] for event in events]

        self.assertIn("tool_result", event_names)
        self.assertIn("answer_delta", event_names)
        self.assertEqual(event_names[-1], "final")
        self.assertLess(event_names.index("status"), event_names.index("tool_result"))
        self.assertLess(event_names.index("tool_result"), event_names.index("answer_delta"))
        self.assertEqual(
            [event["data"]["content"] for event in events if event["event"] == "answer_delta"],
            ["最终", "回答"],
        )
        self.assertEqual(events[-1]["data"]["answer"], "最终回答")
        self.assertEqual(events[-1]["data"]["citations"][0]["page"], 12)


if __name__ == "__main__":
    unittest.main()
