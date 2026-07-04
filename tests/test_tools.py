import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.documents import DocumentService
from app.tools import ListReportsTool, ReadPdfPageTool, SearchReportTextTool
from app.tools import ToolRegistry, extract_text_tool_calls


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_search_document_service(root: Path) -> DocumentService:
    pdf = root / "raw" / "report.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.4")
    for doc_id, pages in {
        "doc-a": ["第一页营收 100 亿元，净利润增长", "第二页净利润 20 亿元"],
        "doc-b": ["第一页营收 50 亿元"],
    }.items():
        artifact = root / "mineru" / doc_id
        write_json(
            artifact / "manifest.json",
            {"doc_id": doc_id, "file_name": "report.pdf", "source_path": str(pdf), "page_count": len(pages)},
        )
        write_json(
            artifact / "content_list_v2.json",
            [
                [{"type": "paragraph", "content": {"paragraph_content": text}}]
                for text in pages
            ],
        )
    return DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")


class FakeTool:
    name = "search"
    aliases = ("tavily_search",)

    def schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    def run(self, arguments):
        return {
            "arguments": arguments,
            "citations": [{"doc_id": "url", "doc_name": "Title", "page": None}],
        }


class ToolRegistryTests(unittest.TestCase):
    def test_schema_exposes_each_tool_once_even_with_aliases(self) -> None:
        registry = ToolRegistry([FakeTool()])

        schemas = registry.schemas()

        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "search")

    def test_execute_supports_tool_alias(self) -> None:
        registry = ToolRegistry([FakeTool()])

        result = registry.execute(
            extract_text_tool_calls(
                "<tool_call><function=tavily_search><parameter=query>Google</parameter></function></tool_call>"
            )[0]
        )

        self.assertEqual(result.status, "done")
        self.assertEqual(result.name, "tavily_search")
        self.assertEqual(result.content["arguments"]["query"], "Google")
        self.assertEqual(result.citations[0]["doc_name"], "Title")

    def test_extract_text_tool_call_normalizes_mimo_parameters(self) -> None:
        calls = extract_text_tool_calls(
            "<tool_call> <function=search> "
            "<parameter=query>Google 最新新闻</parameter>"
            "<parameter=type>news</parameter>"
            "<parameter=limit>10</parameter>"
            "</function> </tool_call>"
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].arguments["query"], "Google 最新新闻")
        self.assertEqual(calls[0].arguments["topic"], "news")
        self.assertEqual(calls[0].arguments["max_results"], 10)

    def test_report_tools_list_and_read_page_with_citation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "raw" / "report.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-1.4")
            artifact = root / "mineru" / "doc-a"
            write_json(
                artifact / "manifest.json",
                {"doc_id": "doc-a", "file_name": "report.pdf", "source_path": str(pdf)},
            )
            write_json(
                artifact / "content_list_v2.json",
                [
                    [
                        {
                            "type": "paragraph",
                            "content": {"paragraph_content": [{"type": "text", "content": "营收 100 亿元"}]},
                        }
                    ]
                ],
            )
            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")

            reports = ListReportsTool(service).run({})
            page = ReadPdfPageTool(service).run({"doc_id": "doc-a", "page": 1})

        self.assertEqual(reports["reports"][0]["doc_id"], "doc-a")
        self.assertEqual(page["text"], "营收 100 亿元")
        self.assertEqual(page["citations"][0]["page"], 1)

    def test_read_page_trims_blocks_with_text_budget(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "raw" / "report.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-1.4")
            artifact = root / "mineru" / "doc-a"
            write_json(
                artifact / "manifest.json",
                {"doc_id": "doc-a", "file_name": "report.pdf", "source_path": str(pdf)},
            )
            write_json(
                artifact / "content_list_v2.json",
                [
                    [
                        {"type": "paragraph", "content": {"paragraph_content": "A" * 900}},
                        {"type": "paragraph", "content": {"paragraph_content": "B" * 900}},
                    ]
                ],
            )
            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")

            page = ReadPdfPageTool(service).run({"doc_id": "doc-a", "page": 1, "max_chars": 1000})

        self.assertTrue(page["truncated"])
        self.assertLessEqual(sum(len(block["text"]) for block in page["blocks"]), 1000)

    def test_search_report_text_finds_page_with_citation(self) -> None:
        with TemporaryDirectory() as directory:
            service = build_search_document_service(Path(directory))

            result = SearchReportTextTool(service).run({"doc_id": "doc-a", "query": "20 亿元"})

        self.assertEqual(result["results"][0]["doc_id"], "doc-a")
        self.assertEqual(result["results"][0]["page"], 2)
        self.assertIn("20 亿元", result["results"][0]["snippet"])
        self.assertGreater(result["results"][0]["score"], 0)
        self.assertEqual(result["results"][0]["matched_terms"], ["20 亿元"])
        self.assertEqual(result["citations"][0]["page"], 2)

    def test_search_report_text_can_search_all_docs_and_cap_results(self) -> None:
        with TemporaryDirectory() as directory:
            service = build_search_document_service(Path(directory))

            result = SearchReportTextTool(service).run({"query": "营收", "max_results": 1})

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["doc_id"], "doc-a")

    def test_search_report_text_searches_all_docs_by_default(self) -> None:
        with TemporaryDirectory() as directory:
            service = build_search_document_service(Path(directory))

            result = SearchReportTextTool(service).run({"query": "营收", "max_results": 5})

        self.assertEqual([item["doc_id"] for item in result["results"]], ["doc-a", "doc-b"])

    def test_search_report_text_ranks_multi_term_matches(self) -> None:
        with TemporaryDirectory() as directory:
            service = build_search_document_service(Path(directory))

            result = SearchReportTextTool(service).run({"query": "营收 净利润", "max_results": 5})

        self.assertEqual(result["results"][0]["doc_id"], "doc-a")
        self.assertEqual(result["results"][0]["page"], 1)
        self.assertIn("营收", result["results"][0]["matched_terms"])

    def test_search_report_text_rejects_blank_query(self) -> None:
        with TemporaryDirectory() as directory:
            service = build_search_document_service(Path(directory))

            with self.assertRaisesRegex(ValueError, "query must not be blank"):
                SearchReportTextTool(service).run({"query": "  "})


if __name__ == "__main__":
    unittest.main()
