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

    def test_run_turn_builds_citations_from_table_tools(self) -> None:
        llm_client = MagicMock()
        llm_client.generate.side_effect = [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="search_tables",
                        arguments={"query": "营业收入"},
                        tool_call_id="call-1",
                    ),
                    ToolCall(
                        tool_name="get_table",
                        arguments={"table_id": "table-1"},
                        tool_call_id="call-2",
                    ),
                ],
            ),
            AssistantMessage(content="最终回答"),
        ]

        tool_registry = MagicMock()
        tool_registry.execute.side_effect = [
            {
                "tables": [
                    {
                        "doc_id": "doc-a",
                        "doc_name": "doc-a.pdf",
                        "page_start": 5,
                        "table_id": "table-1",
                    }
                ]
            },
            {
                "table": {
                    "doc_id": "doc-a",
                    "doc_name": "doc-a.pdf",
                    "page_start": 5,
                    "table_id": "table-1",
                }
            },
        ]

        runtime = SingleAgentRuntime(
            llm_client=llm_client,
            tool_registry=tool_registry,
            session_store=InMemorySessionStore(),
        )

        result = runtime.run_turn("营业收入是多少？", session_id="s1")

        self.assertEqual(result.citations[0].doc_id, "doc-a")
        self.assertEqual(result.citations[0].page, 5)
        self.assertEqual(len(result.citations), 1)

    def test_run_turn_strips_chart_option_json_from_final_answer(self) -> None:
        llm_client = MagicMock()
        llm_client.generate.side_effect = [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        tool_name="create_chart",
                        arguments={
                            "chart_type": "bar",
                            "title": "营业收入对比",
                            "categories": ["2023年度", "2024年度"],
                            "series": [{"name": "营业收入", "values": [1476.94, 1708.99]}],
                        },
                        tool_call_id="call-chart",
                    )
                ],
            ),
            AssistantMessage(
                content=(
                    "根据年报数据，我为你绘制了营业收入对比图表：\n\n"
                    "{\n"
                    '  "title": {"text": "贵州茅台2023-2024年度营业收入对比"},\n'
                    '  "tooltip": {"trigger": "axis"},\n'
                    '  "xAxis": {"type": "category", "data": ["2023年度", "2024年度"]},\n'
                    '  "yAxis": {"type": "value", "name": "亿元"},\n'
                    '  "series": [{"name": "营业收入", "type": "bar", "data": [1476.94, 1708.99]}]\n'
                    "}\n\n"
                    "数据说明：2024 年营业收入为 1,708.99 亿元。"
                )
            ),
        ]
        tool_registry = MagicMock()
        tool_registry.execute.return_value = {
            "chart_id": "chart-1",
            "chart_type": "bar",
            "echarts_option": {"series": [{"type": "bar"}]},
        }

        runtime = SingleAgentRuntime(
            llm_client=llm_client,
            tool_registry=tool_registry,
            session_store=InMemorySessionStore(),
        )

        result = runtime.run_turn("画一下营业收入对比图", session_id="s1")

        self.assertIn("数据说明", result.answer)
        self.assertNotIn('"series"', result.answer)
        self.assertNotIn('"xAxis"', result.answer)


if __name__ == "__main__":
    unittest.main()
