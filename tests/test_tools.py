import unittest
from unittest.mock import MagicMock

from app.domain import Evidence
from app.tools import (
    ToolRegistry,
    build_search_reports_tool,
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
        registry = ToolRegistry(
            [
                build_search_reports_tool(retriever),
            ]
        )

        definitions = registry.get_definitions()
        names = [item["function"]["name"] for item in definitions]

        self.assertEqual(names, ["search_reports"])
        registry.execute("search_reports", query="营业总收入是多少？")
        retriever.search.assert_called_once_with("营业总收入是多少？", top_k=3, filters=None)


if __name__ == "__main__":
    unittest.main()
