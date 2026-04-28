import unittest
from unittest.mock import MagicMock

from app.domain import Evidence
from app.tools import (
    ToolRegistry,
    build_create_chart_tool,
    build_default_tool_registry,
    build_get_table_tool,
    build_search_reports_tool,
    build_search_tables_tool,
)


class ToolTests(unittest.TestCase):
    def test_search_reports_executes_retriever_and_returns_normalized_payload(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = [
            Evidence(
                chunk_id="doc-a-page-1-chunk-1",
                doc_id="doc-a",
                doc_name="doc-a.pdf",
                source_path="/tmp/doc-a.pdf",
                page=1,
                page_start=1,
                page_end=1,
                chunk_type="table",
                section_path=["第一章", "财务摘要"],
                text="收入增长",
                score=0.91,
            )
        ]
        retriever.get_last_retrieval_queries.return_value = ["营业总收入是多少？", "主要会计数据 营业总收入"]
        tool = build_search_reports_tool(retriever)

        result = tool.execute(query="营业总收入是多少？", top_k=2, doc_id="doc-a")

        retriever.search.assert_called_once_with("营业总收入是多少？", top_k=2, filters={"doc_id": "doc-a"})
        self.assertEqual(result["query"], "营业总收入是多少？")
        self.assertEqual(
            result["results"],
            [
                {
                    "doc_id": "doc-a",
                    "doc_name": "doc-a.pdf",
                    "page": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_type": "table",
                    "section_path": ["第一章", "财务摘要"],
                    "text": "收入增长",
                    "score": 0.91,
                }
            ],
        )
        self.assertEqual(result["retrieval_queries"], ["营业总收入是多少？", "主要会计数据 营业总收入"])

    def test_search_reports_returns_empty_results(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = []
        tool = build_search_reports_tool(retriever)

        result = tool.execute(query="营业总收入是多少？")

        self.assertEqual(
            result,
            {
                "query": "营业总收入是多少？",
                "retrieval_queries": ["营业总收入是多少？"],
                "results": [],
            },
        )

    def test_search_reports_supports_multiple_document_filters(self) -> None:
        retriever = MagicMock()
        retriever.search.return_value = []
        tool = build_search_reports_tool(retriever)

        tool.execute(query="对比营收", top_k=4, doc_ids=["doc-a", "doc-b"])

        retriever.search.assert_called_once_with(
            "对比营收",
            top_k=4,
            filters={"doc_id": {"$in": ["doc-a", "doc-b"]}},
        )

    def test_tool_registry_exposes_tool_definition_and_executes_named_tool(self) -> None:
        retriever = MagicMock()
        table_repository = MagicMock()
        registry = ToolRegistry(
            [
                build_search_reports_tool(retriever),
                build_search_tables_tool(table_repository),
                build_get_table_tool(table_repository),
                build_create_chart_tool(),
            ]
        )

        definitions = registry.get_definitions()
        names = [item["function"]["name"] for item in definitions]

        self.assertEqual(names, ["search_reports", "search_tables", "get_table", "create_chart"])
        registry.execute("search_reports", query="营业总收入是多少？")
        retriever.search.assert_called_once_with("营业总收入是多少？", top_k=3, filters=None)

    def test_default_tool_registry_includes_search_and_chart_tools(self) -> None:
        registry = build_default_tool_registry(MagicMock(), table_repository=MagicMock())

        names = [item["function"]["name"] for item in registry.get_definitions()]

        self.assertEqual(names, ["search_reports", "search_tables", "get_table", "create_chart"])

    def test_search_tables_executes_table_repository(self) -> None:
        table_repository = MagicMock()
        table_repository.search_tables.return_value = [
            {
                "table_id": "table-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "title": "主要会计数据",
                "page_start": 5,
                "page_end": 5,
                "preview_matrix": [["指标", "2024"], ["营业收入", "100"]],
                "score": 3.2,
            }
        ]
        tool = build_search_tables_tool(table_repository)

        result = tool.execute(
            query="营业收入",
            top_k=2,
            doc_ids=["doc-a", "doc-b"],
            statement_type="key_metrics",
        )

        table_repository.search_tables.assert_called_once_with(
            query="营业收入",
            top_k=2,
            doc_ids=["doc-a", "doc-b"],
            statement_type="key_metrics",
        )
        self.assertEqual(result["tables"][0]["table_id"], "table-1")
        self.assertEqual(result["doc_ids"], ["doc-a", "doc-b"])

    def test_get_table_executes_table_repository(self) -> None:
        table_repository = MagicMock()
        table_repository.get_table.return_value = {
            "table_id": "table-1",
            "doc_id": "doc-a",
            "doc_name": "doc-a.pdf",
            "matrix": [["指标", "2024"], ["营业收入", "100"]],
        }
        tool = build_get_table_tool(table_repository)

        result = tool.execute(table_id="table-1", doc_id="doc-a")

        table_repository.get_table.assert_called_once_with(table_id="table-1", doc_id="doc-a")
        self.assertEqual(result["table"]["matrix"][1][0], "营业收入")
        self.assertIsNone(result["error"])

    def test_create_chart_generates_grouped_bar_option(self) -> None:
        tool = build_create_chart_tool()

        result = tool.execute(
            chart_type="grouped_bar",
            title="2024 年营收利润对比",
            categories=["营业收入", "归母净利润"],
            series=[
                {"name": "贵州茅台", "values": [1709.0, 862.28]},
                {"name": "长江电力", "values": [844.92, 324.96]},
            ],
            unit="亿元",
            source_notes=["茅台 2024 年报第 5 页", "长江电力 2024 年报第 5 页"],
        )

        self.assertEqual(result["chart_type"], "grouped_bar")
        self.assertTrue(result["chart_id"].startswith("chart-"))
        self.assertEqual(result["source_notes"], ["茅台 2024 年报第 5 页", "长江电力 2024 年报第 5 页"])
        option = result["echarts_option"]
        self.assertEqual(option["xAxis"]["data"], ["营业收入", "归母净利润"])
        self.assertEqual(option["yAxis"]["name"], "亿元")
        self.assertEqual(option["series"][0]["type"], "bar")
        self.assertEqual(option["series"][1]["data"], [844.92, 324.96])

    def test_create_chart_generates_line_combo_and_pie_options(self) -> None:
        tool = build_create_chart_tool()

        line = tool.execute(
            chart_type="line",
            title="营收趋势",
            categories=["2022", "2023", "2024"],
            series=[{"name": "营业收入", "values": [100.0, 120.0, 150.0]}],
        )
        combo = tool.execute(
            chart_type="combo",
            title="营收与增速",
            categories=["2022", "2023", "2024"],
            series=[
                {"name": "营业收入", "values": [100.0, 120.0, 150.0], "type": "bar", "unit": "亿元"},
                {"name": "增速", "values": [5.0, 20.0, 25.0], "type": "line", "unit": "%", "y_axis": "right"},
            ],
        )
        pie = tool.execute(
            chart_type="pie",
            title="收入结构",
            categories=["产品 A", "产品 B"],
            series=[{"name": "收入", "values": [70.0, 30.0]}],
        )

        self.assertEqual(line["echarts_option"]["series"][0]["type"], "line")
        self.assertEqual(combo["echarts_option"]["series"][1]["yAxisIndex"], 1)
        self.assertEqual(combo["echarts_option"]["yAxis"][1]["name"], "%")
        self.assertEqual(pie["echarts_option"]["series"][0]["type"], "pie")
        self.assertEqual(pie["echarts_option"]["series"][0]["data"][0], {"name": "产品 A", "value": 70.0})

    def test_create_chart_rejects_invalid_lengths_and_values(self) -> None:
        tool = build_create_chart_tool()

        with self.assertRaisesRegex(ValueError, "length must match"):
            tool.execute(
                chart_type="bar",
                title="坏图",
                categories=["营业收入", "归母净利润"],
                series=[{"name": "贵州茅台", "values": [1709.0]}],
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            tool.execute(
                chart_type="bar",
                title="坏图",
                categories=["营业收入"],
                series=[{"name": "贵州茅台", "values": [float("nan")]}],
            )

    def test_create_chart_rejects_oversized_inputs(self) -> None:
        tool = build_create_chart_tool()

        with self.assertRaisesRegex(ValueError, "at most 24"):
            tool.execute(
                chart_type="bar",
                title="太多分类",
                categories=[str(index) for index in range(25)],
                series=[{"name": "指标", "values": [float(index) for index in range(25)]}],
            )
        with self.assertRaisesRegex(ValueError, "at most 8"):
            tool.execute(
                chart_type="bar",
                title="太多系列",
                categories=["2024"],
                series=[{"name": f"公司{index}", "values": [1.0]} for index in range(9)],
            )


if __name__ == "__main__":
    unittest.main()
