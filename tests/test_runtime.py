import unittest
from unittest.mock import MagicMock

from app.domain import ConversationState
from app.messages import AssistantMessage, ToolCall
from app.runtime import SingleAgentRuntime
from app.session import InMemorySessionStore


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


if __name__ == "__main__":
    unittest.main()
