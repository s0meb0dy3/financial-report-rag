import unittest
from unittest.mock import MagicMock

from app.domain import DocumentRef, Evidence
from app.tools import (
    ToolRegistry,
    build_extract_table_tool,
    build_list_reports_tool,
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

    def test_list_reports_returns_unique_sorted_documents(self) -> None:
        retriever = MagicMock()
        retriever.list_documents.return_value = [
            DocumentRef(doc_id="pingan", doc_name="中国平安2024年年度报告.pdf"),
            DocumentRef(doc_id="moutai", doc_name="贵州茅台2024年年度报告.pdf"),
        ]
        tool = build_list_reports_tool(retriever)

        result = tool.execute()

        retriever.list_documents.assert_called_once_with()
        self.assertEqual(
            result,
            {
                "documents": [
                    {"doc_id": "pingan", "doc_name": "中国平安2024年年度报告.pdf"},
                    {"doc_id": "moutai", "doc_name": "贵州茅台2024年年度报告.pdf"},
                ]
            },
        )

    def test_search_tables_returns_candidate_tables(self) -> None:
        table_repository = MagicMock()
        table_repository.search_tables.return_value = [
            {
                "table_id": "doc-a-logical-table-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "title": "合并现金流量表",
                "statement_type_guess": "cash_flow",
                "section_path": ["财务报告"],
                "page_start": 10,
                "page_end": 11,
                "preview_matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                "score": 4.2,
            }
        ]
        tool = build_search_tables_tool(table_repository)

        result = tool.execute(
            doc_id="doc-a",
            query="经营活动产生的现金流量净额",
            statement_type="cash_flow",
            top_k=2,
        )

        table_repository.search_tables.assert_called_once_with(
            doc_id="doc-a",
            query="经营活动产生的现金流量净额",
            statement_type="cash_flow",
            top_k=2,
        )
        self.assertEqual(result["doc_id"], "doc-a")
        self.assertEqual(result["tables"][0]["table_id"], "doc-a-logical-table-1")

    def test_search_tables_validates_required_inputs(self) -> None:
        table_repository = MagicMock()
        tool = build_search_tables_tool(table_repository)

        result = tool.execute(doc_id="doc-a")

        self.assertEqual(result["doc_id"], "doc-a")
        self.assertEqual(result["tables"], [])
        self.assertIn("query or statement_type is required", result["error"])
        table_repository.search_tables.assert_not_called()

    def test_extract_table_returns_full_matrix(self) -> None:
        table_repository = MagicMock()
        table_repository.get_table.return_value = {
            "table_id": "doc-a-logical-table-1",
            "doc_id": "doc-a",
            "doc_name": "doc-a.pdf",
            "title": "主要会计数据",
            "statement_type_guess": "key_metrics",
            "section_path": ["第一章", "主要会计数据"],
            "page_start": 2,
            "page_end": 2,
            "matrix": [["项目", "2024年"], ["营业总收入", "100"]],
            "footnotes_text": "单位：元",
            "fragments": [{"source_element_id": "table-1", "page_start": 2, "page_end": 2, "row_count": 2}],
            "row_count": 2,
            "column_count": 2,
        }
        tool = build_extract_table_tool(table_repository)

        result = tool.execute(doc_id="doc-a", table_id="doc-a-logical-table-1")

        table_repository.get_table.assert_called_once_with(doc_id="doc-a", table_id="doc-a-logical-table-1")
        self.assertEqual(result["table"]["table_id"], "doc-a-logical-table-1")
        self.assertEqual(result["table"]["matrix"][1][0], "营业总收入")

    def test_tool_registry_exposes_tool_definition_and_executes_named_tool(self) -> None:
        retriever = MagicMock()
        table_repository = MagicMock()
        registry = ToolRegistry(
            [
                build_search_tables_tool(table_repository),
                build_extract_table_tool(table_repository),
                build_search_reports_tool(retriever),
                build_list_reports_tool(retriever),
            ]
        )

        definitions = registry.get_definitions()
        names = [item["function"]["name"] for item in definitions]

        self.assertEqual(names, ["search_tables", "extract_table", "search_reports", "list_reports"])
        registry.execute("search_tables", doc_id="doc-a", statement_type="cash_flow")
        registry.execute("extract_table", doc_id="doc-a", table_id="doc-a-logical-table-1")
        registry.execute("search_reports", query="营业总收入是多少？")
        registry.execute("list_reports")
        table_repository.search_tables.assert_called_once_with(
            doc_id="doc-a",
            query=None,
            statement_type="cash_flow",
            top_k=5,
        )
        table_repository.get_table.assert_called_once_with(doc_id="doc-a", table_id="doc-a-logical-table-1")
        retriever.search.assert_called_once_with("营业总收入是多少？", top_k=3, filters=None)
        retriever.list_documents.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
