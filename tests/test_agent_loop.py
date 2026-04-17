import json
import os
import unittest
from unittest.mock import MagicMock, patch

from app.agent import AgentLoop, DEFAULT_CHAT_MODEL, MAX_TOOL_CALLS


def make_message(*, content: str | None = None, tool_calls: list | None = None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    return message


def make_response(message) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


class AgentLoopTests(unittest.TestCase):
    def test_default_chat_model_uses_qwen_3_6_plus_free(self) -> None:
        self.assertEqual(DEFAULT_CHAT_MODEL, "qwen/qwen3.6-plus:free")

    def test_from_env_reads_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            loop = AgentLoop.from_env()

        self.assertEqual(loop.api_key, "test-key")
        self.assertEqual(loop.chat_model, DEFAULT_CHAT_MODEL)

    def test_run_turn_returns_answer_without_tool_calls(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response(make_message(content="直接回答"))

        tool_registry = MagicMock()
        loop = AgentLoop(api_key="test-key", client=client, tool_registry=tool_registry)

        result = loop.run_turn("营业总收入是多少？")

        self.assertEqual(result["answer"], "直接回答")
        self.assertEqual(result["tool_results"], [])
        tool_registry.execute.assert_not_called()

    def test_run_turn_executes_tool_and_feeds_result_back(self) -> None:
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "search_reports"
        tool_call.function.arguments = json.dumps({"query": "营业总收入是多少？", "top_k": 2}, ensure_ascii=False)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(content="基于检索结果的最终回答")),
        ]

        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = [{"type": "function", "function": {"name": "search_reports"}}]
        tool_registry.execute.return_value = {
            "query": "营业总收入是多少？",
            "results": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2, "text": "收入", "score": 0.91}],
        }
        loop = AgentLoop(api_key="test-key", client=client, tool_registry=tool_registry)

        result = loop.run_turn("营业总收入是多少？")

        self.assertEqual(result["answer"], "基于检索结果的最终回答")
        self.assertEqual(
            result["citations"],
            [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2}],
        )
        self.assertEqual(len(result["tool_results"]), 1)
        self.assertEqual(result["tool_results"][0]["tool_name"], "search_reports")
        self.assertEqual(result["tool_results"][0]["arguments"], {"query": "营业总收入是多少？", "top_k": 2})
        tool_registry.execute.assert_called_once_with("search_reports", query="营业总收入是多少？", top_k=2)

        second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(second_call_messages[-1]["role"], "tool")
        self.assertIn("doc-a.pdf", second_call_messages[-1]["content"])

    def test_run_turn_stops_after_max_tool_calls_and_requests_final_answer(self) -> None:
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "search_reports"
        tool_call.function.arguments = json.dumps({"query": "营业总收入是多少？"}, ensure_ascii=False)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(content="请基于已有证据给出最终答案")),
        ]

        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = [{"type": "function", "function": {"name": "search_reports"}}]
        tool_registry.execute.return_value = {"query": "营业总收入是多少？", "results": []}
        loop = AgentLoop(api_key="test-key", client=client, tool_registry=tool_registry)

        result = loop.run_turn("营业总收入是多少？")

        self.assertEqual(len(result["tool_results"]), MAX_TOOL_CALLS)
        self.assertEqual(result["answer"], "请基于已有证据给出最终答案")
        self.assertEqual(client.chat.completions.create.call_count, MAX_TOOL_CALLS + 1)

        final_call_messages = client.chat.completions.create.call_args_list[-1].kwargs["messages"]
        self.assertEqual(final_call_messages[-1]["role"], "user")
        self.assertIn("不要继续调用工具", final_call_messages[-1]["content"])

    def test_run_turn_uses_instance_defaults_for_search_reports(self) -> None:
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "search_reports"
        tool_call.function.arguments = json.dumps({"query": "营业总收入是多少？"}, ensure_ascii=False)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(content="默认参数回答")),
        ]

        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = [{"type": "function", "function": {"name": "search_reports"}}]
        tool_registry.execute.return_value = {"query": "营业总收入是多少？", "results": []}
        loop = AgentLoop(api_key="test-key", client=client, tool_registry=tool_registry, top_k=4, doc_id="doc-a")

        loop.run_turn("营业总收入是多少？")

        tool_registry.execute.assert_called_once_with(
            "search_reports",
            query="营业总收入是多少？",
            top_k=4,
            doc_id="doc-a",
        )

    def test_run_turn_allows_turn_level_search_override(self) -> None:
        tool_call = MagicMock()
        tool_call.id = "call-1"
        tool_call.function.name = "search_reports"
        tool_call.function.arguments = json.dumps({"query": "营业总收入是多少？"}, ensure_ascii=False)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_response(make_message(tool_calls=[tool_call])),
            make_response(make_message(content="覆盖参数回答")),
        ]

        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = [{"type": "function", "function": {"name": "search_reports"}}]
        tool_registry.execute.return_value = {"query": "营业总收入是多少？", "results": []}
        loop = AgentLoop(api_key="test-key", client=client, tool_registry=tool_registry, top_k=3, doc_id="default")

        loop.run_turn("营业总收入是多少？", top_k=5, doc_id="doc-b")

        tool_registry.execute.assert_called_once_with(
            "search_reports",
            query="营业总收入是多少？",
            top_k=5,
            doc_id="doc-b",
        )

    def test_run_turn_prefers_table_tools_and_collects_table_citations(self) -> None:
        search_tables_call = MagicMock()
        search_tables_call.id = "call-1"
        search_tables_call.function.name = "search_tables"
        search_tables_call.function.arguments = json.dumps(
            {"query": "经营活动产生的现金流量净额", "statement_type": "cash_flow"},
            ensure_ascii=False,
        )
        extract_table_call = MagicMock()
        extract_table_call.id = "call-2"
        extract_table_call.function.name = "extract_table"
        extract_table_call.function.arguments = json.dumps({"table_id": "doc-a-logical-table-1"}, ensure_ascii=False)

        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_response(make_message(tool_calls=[search_tables_call])),
            make_response(make_message(tool_calls=[extract_table_call])),
            make_response(make_message(content="最终回答")),
        ]

        tool_registry = MagicMock()
        tool_registry.get_definitions.return_value = [
            {"type": "function", "function": {"name": "search_tables"}},
            {"type": "function", "function": {"name": "extract_table"}},
        ]
        tool_registry.execute.side_effect = [
            {
                "doc_id": "doc-a",
                "tables": [
                    {
                        "table_id": "doc-a-logical-table-1",
                        "doc_id": "doc-a",
                        "doc_name": "doc-a.pdf",
                        "title": "合并现金流量表",
                        "page_start": 10,
                        "page_end": 11,
                        "section_path": ["财务报告"],
                        "preview_matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                        "score": 4.2,
                    }
                ],
            },
            {
                "doc_id": "doc-a",
                "table": {
                    "table_id": "doc-a-logical-table-1",
                    "doc_id": "doc-a",
                    "doc_name": "doc-a.pdf",
                    "title": "合并现金流量表",
                    "page_start": 10,
                    "page_end": 11,
                    "section_path": ["财务报告"],
                    "matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                    "footnotes_text": "",
                    "fragments": [{"source_element_id": "table-1", "page_start": 10, "page_end": 11, "row_count": 2}],
                    "row_count": 2,
                    "column_count": 2,
                },
            },
        ]
        loop = AgentLoop(api_key="test-key", client=client, tool_registry=tool_registry, doc_id="doc-a")

        result = loop.run_turn("经营活动产生的现金流量净额是多少？")

        self.assertEqual(result["answer"], "最终回答")
        self.assertEqual(
            result["citations"],
            [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 10}],
        )
        self.assertEqual(
            tool_registry.execute.call_args_list[0].args,
            ("search_tables",),
        )
        self.assertEqual(
            tool_registry.execute.call_args_list[0].kwargs,
            {"query": "经营活动产生的现金流量净额", "statement_type": "cash_flow", "doc_id": "doc-a"},
        )
        self.assertEqual(
            tool_registry.execute.call_args_list[1].kwargs,
            {"table_id": "doc-a-logical-table-1", "doc_id": "doc-a"},
        )


if __name__ == "__main__":
    unittest.main()
