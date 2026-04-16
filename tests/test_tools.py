import unittest
from unittest.mock import MagicMock

from app.domain import DocumentRef, Evidence
from app.tools import ToolRegistry, build_list_reports_tool, build_search_reports_tool


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

    def test_tool_registry_exposes_tool_definition_and_executes_named_tool(self) -> None:
        retriever = MagicMock()
        registry = ToolRegistry([build_search_reports_tool(retriever), build_list_reports_tool(retriever)])

        definitions = registry.get_definitions()

        self.assertEqual(definitions[0]["function"]["name"], "search_reports")
        self.assertEqual(definitions[1]["function"]["name"], "list_reports")
        registry.execute("search_reports", query="营业总收入是多少？")
        registry.execute("list_reports")
        retriever.search.assert_called_once_with("营业总收入是多少？", top_k=3, filters=None)
        retriever.list_documents.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
