import json
import tempfile
import unittest
from pathlib import Path
from shutil import copy2
from unittest.mock import MagicMock, patch

from app.ingestion import (
    ChunkRecord,
    IngestionArtifacts,
    IngestionService,
    ParsedDocument,
    StructuredDoclingChunker,
    chunk_page_text,
    discover_pdf_files,
    ingest_pdfs,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PDF_PATH = RAW_DIR / "茅台2024年年度报告完整版.pdf"
SECOND_PDF_PATH = RAW_DIR / "长江电力2024年报.pdf"


class IngestPdfTests(unittest.TestCase):
    def test_structured_docling_chunker_groups_sections_and_tables(self) -> None:
        parsed_document = ParsedDocument(
            doc_id="doc-a",
            doc_name="doc-a.pdf",
            source_path="/tmp/doc-a.pdf",
            raw_doc={"schema_name": "DoclingDocument"},
            elements=[
                {
                    "element_id": "heading-1",
                    "kind": "heading",
                    "text": "第一章 经营情况",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "para-1",
                    "kind": "paragraph",
                    "text": "本期公司营业收入稳步增长。",
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "table-1",
                    "kind": "table",
                    "text": "| 项目 | 金额 |\n| --- | --- |\n| 营业收入 | 100 |",
                    "page_start": 2,
                    "page_end": 2,
                    "provenance": [{"page": 2}],
                },
            ],
        )

        chunks = StructuredDoclingChunker(max_chars=200).chunk(parsed_document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_type"], "paragraph")
        self.assertEqual(chunks[0]["section_path"], ["第一章 经营情况"])
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertEqual(chunks[0]["page_end"], 1)
        self.assertEqual(chunks[0]["element_ids"], ["para-1"])
        self.assertEqual(chunks[1]["chunk_type"], "table")
        self.assertEqual(chunks[1]["section_path"], ["第一章 经营情况"])
        self.assertEqual(chunks[1]["page_start"], 2)
        self.assertEqual(chunks[1]["page_end"], 2)
        self.assertEqual(chunks[1]["element_ids"], ["table-1"])

    def test_structured_docling_chunker_splits_long_section_and_preserves_provenance(self) -> None:
        parsed_document = ParsedDocument(
            doc_id="doc-b",
            doc_name="doc-b.pdf",
            source_path="/tmp/doc-b.pdf",
            raw_doc={"schema_name": "DoclingDocument"},
            elements=[
                {
                    "element_id": "heading-1",
                    "kind": "heading",
                    "text": "第二章 财务摘要",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "para-1",
                    "kind": "paragraph",
                    "text": "第一段。" * 80,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "para-2",
                    "kind": "paragraph",
                    "text": "第二段。" * 80,
                    "page_start": 2,
                    "page_end": 2,
                    "provenance": [{"page": 2}],
                },
            ],
        )

        chunks = StructuredDoclingChunker(max_chars=240).chunk(parsed_document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["section_path"] == ["第二章 财务摘要"] for chunk in chunks))
        self.assertTrue(all(chunk["provenance"] for chunk in chunks))
        self.assertEqual(chunks[0]["page_start"], 1)
        self.assertEqual(chunks[-1]["page_end"], 2)

    def test_chunk_page_text_splits_long_text_into_multiple_chunks(self) -> None:
        text = "第一段。" * 120 + "\n" + "第二段。" * 120

        chunks = chunk_page_text(text, page_number=1, max_chars=300, overlap_chars=60)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))

    def test_discover_pdf_files_reads_all_pdfs_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            copy2(PDF_PATH, raw_dir / PDF_PATH.name)
            copy2(SECOND_PDF_PATH, raw_dir / SECOND_PDF_PATH.name)

            discovered = discover_pdf_files(raw_dir)

        self.assertEqual(discovered, sorted([raw_dir / PDF_PATH.name, raw_dir / SECOND_PDF_PATH.name]))

    def test_ingest_pdfs_writes_multi_document_output(self) -> None:
        first_doc = ParsedDocument(
            doc_id="doc-a",
            doc_name=PDF_PATH.name,
            source_path=str(PDF_PATH),
            raw_doc={"schema_name": "DoclingDocument", "name": PDF_PATH.name},
            markdown="# 贵州茅台",
            elements=[
                {
                    "element_id": "heading-1",
                    "kind": "heading",
                    "text": "贵州茅台",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "para-1",
                    "kind": "paragraph",
                    "text": "营业收入增长。",
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
            ],
        )
        second_doc = ParsedDocument(
            doc_id="doc-b",
            doc_name=SECOND_PDF_PATH.name,
            source_path=str(SECOND_PDF_PATH),
            raw_doc={"schema_name": "DoclingDocument", "name": SECOND_PDF_PATH.name},
            markdown="# 长江电力",
            elements=[
                {
                    "element_id": "heading-1",
                    "kind": "heading",
                    "text": "长江电力",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "para-1",
                    "kind": "paragraph",
                    "text": "来水偏丰。",
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "table-1",
                    "kind": "table",
                    "text": "| 指标 | 数值 |\n| --- | --- |\n| 发电量 | 100 |",
                    "page_start": 2,
                    "page_end": 2,
                    "provenance": [{"page": 2}],
                },
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chunks.json"

            with patch("app.ingestion.service.DoclingPdfParser") as parser_cls:
                parser_cls.return_value.parse.side_effect = [first_doc, second_doc]
                chunks = ingest_pdfs([PDF_PATH, SECOND_PDF_PATH], output_path)

            self.assertGreaterEqual(len(chunks), 3)
            self.assertTrue(output_path.exists())

            saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved_chunks, chunks)
            self.assertEqual(saved_chunks[0]["page_start"], 1)
            self.assertTrue(all(chunk["chunk_id"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["text"].strip() for chunk in saved_chunks))
            self.assertTrue(all(chunk["doc_id"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["doc_name"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["source_path"] for chunk in saved_chunks))
            self.assertEqual({chunk["doc_name"] for chunk in saved_chunks}, {PDF_PATH.name, SECOND_PDF_PATH.name})
            self.assertTrue(
                all(Path(chunk["source_path"]).name == chunk["doc_name"] for chunk in saved_chunks)
            )

    def test_ingestion_service_writes_docling_json_markdown_and_chunks(self) -> None:
        parsed_document = ParsedDocument(
            doc_id="doc-c",
            doc_name="doc-c.pdf",
            source_path="/tmp/doc-c.pdf",
            raw_doc={"schema_name": "DoclingDocument", "name": "doc-c"},
            markdown="# 第一章\n\n收入增长",
            elements=[
                {
                    "element_id": "heading-1",
                    "kind": "heading",
                    "text": "第一章",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "para-1",
                    "kind": "paragraph",
                    "text": "收入增长",
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
            ],
        )
        parser = MagicMock()
        parser.parse.return_value = parsed_document
        service = IngestionService(parser=parser, chunk_strategy=StructuredDoclingChunker(max_chars=200))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = IngestionArtifacts(
                chunks_path=Path(temp_dir) / "chunks.json",
                docling_json_dir=Path(temp_dir) / "docling",
                markdown_dir=Path(temp_dir) / "markdown",
            )

            chunks = service.ingest_pdfs([Path("/tmp/doc-c.pdf")], artifacts)

            saved_chunks = json.loads(artifacts.chunks_path.read_text(encoding="utf-8"))
            saved_doc = json.loads((artifacts.docling_json_dir / "doc-c.json").read_text(encoding="utf-8"))
            saved_markdown = (artifacts.markdown_dir / "doc-c.md").read_text(encoding="utf-8")

        parser.parse.assert_called_once()
        self.assertEqual(saved_chunks, chunks)
        self.assertEqual(saved_doc["schema_name"], "DoclingDocument")
        self.assertEqual(saved_markdown, "# 第一章\n\n收入增长")

    def test_ingestion_service_can_disable_markdown_export(self) -> None:
        parsed_document = ParsedDocument(
            doc_id="doc-d",
            doc_name="doc-d.pdf",
            source_path="/tmp/doc-d.pdf",
            raw_doc={"schema_name": "DoclingDocument"},
            markdown="# 标题",
            elements=[
                {
                    "element_id": "heading-1",
                    "kind": "heading",
                    "text": "标题",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                }
            ],
        )
        parser = MagicMock()
        parser.parse.return_value = parsed_document
        service = IngestionService(parser=parser, chunk_strategy=StructuredDoclingChunker(max_chars=200))

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = IngestionArtifacts(
                chunks_path=Path(temp_dir) / "chunks.json",
                docling_json_dir=Path(temp_dir) / "docling",
                markdown_dir=Path(temp_dir) / "markdown",
                export_markdown=False,
            )

            service.ingest_pdfs([Path("/tmp/doc-d.pdf")], artifacts)

            markdown_path = artifacts.markdown_dir / "doc-d.md"

        self.assertFalse(markdown_path.exists())

    def test_ingestion_service_raises_when_parser_fails(self) -> None:
        parser = MagicMock()
        parser.parse.side_effect = RuntimeError("docling failed")
        service = IngestionService(parser=parser, chunk_strategy=MagicMock())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = IngestionArtifacts(
                chunks_path=Path(temp_dir) / "chunks.json",
                docling_json_dir=Path(temp_dir) / "docling",
                markdown_dir=Path(temp_dir) / "markdown",
            )

            with self.assertRaisesRegex(RuntimeError, "docling failed"):
                service.ingest_pdfs([Path("/tmp/failed.pdf")], artifacts)


if __name__ == "__main__":
    unittest.main()
