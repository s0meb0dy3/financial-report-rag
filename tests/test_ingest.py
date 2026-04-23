import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from app.ingestion import (
    IngestionArtifacts,
    IngestionService,
    MineruPdfParser,
    ParsedDocument,
    StructuredMineruChunker,
    build_doc_id,
    discover_pdf_files,
    ingest_pdfs,
)


def _write_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")


def _build_zip_bytes(*, pages=None, legacy_items=None, markdown="# Demo", layout=None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        if pages is not None:
            archive.writestr("content_list_v2.json", json.dumps(pages, ensure_ascii=False))
        if legacy_items is not None:
            archive.writestr("demo_content_list.json", json.dumps(legacy_items, ensure_ascii=False))
        archive.writestr("full.md", markdown)
        archive.writestr("layout.json", json.dumps(layout or {"pdf_info": []}, ensure_ascii=False))
    return payload.getvalue()


def _write_cached_artifacts(
    parser: MineruPdfParser,
    pdf_path: Path,
    *,
    pages=None,
    legacy_items=None,
    markdown="# Cached",
) -> Path:
    doc_id = build_doc_id(pdf_path.resolve())
    artifact_dir = parser.artifact_root / doc_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = parser._build_manifest(pdf_path.resolve(), doc_id)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if pages is not None:
        (artifact_dir / "content_list_v2.json").write_text(
            json.dumps(pages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if legacy_items is not None:
        (artifact_dir / "demo_content_list.json").write_text(
            json.dumps(legacy_items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (artifact_dir / "full.md").write_text(markdown, encoding="utf-8")
    return artifact_dir


class MineruParserTests(unittest.TestCase):
    def test_parser_downloads_and_extracts_precision_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "report.pdf"
            _write_pdf(pdf_path)
            artifact_root = root / "artifacts"
            batch_id = "batch-1"
            upload_url = "https://upload.example/report.pdf"
            zip_url = "https://cdn.example/result.zip"
            zip_bytes = _build_zip_bytes(
                pages=[
                    [
                        {
                            "type": "page_header",
                            "content": {"page_header_content": [{"type": "text", "content": "页眉"}]},
                            "bbox": [0, 0, 1, 1],
                        },
                        {
                            "type": "title",
                            "content": {
                                "title_content": [{"type": "text", "content": "第一节 管理层讨论与分析"}],
                                "level": 1,
                            },
                            "bbox": [1, 1, 2, 2],
                        },
                        {
                            "type": "paragraph",
                            "content": {
                                "paragraph_content": [{"type": "text", "content": "营业收入稳步增长。"}]
                            },
                            "bbox": [2, 2, 3, 3],
                        },
                    ],
                    [
                        {
                            "type": "table",
                            "content": {
                                "html": "<table><tr><td>项目</td><td>金额</td></tr><tr><td>营业收入</td><td>100</td></tr></table>",
                                "table_caption": [{"type": "text", "content": "主要会计数据"}],
                                "table_footnote": [{"type": "text", "content": "单位：元"}],
                            },
                            "bbox": [3, 3, 4, 4],
                        },
                        {
                            "type": "page_number",
                            "content": {"page_number_content": [{"type": "text", "content": "2 / 2"}]},
                            "bbox": [4, 4, 5, 5],
                        },
                    ],
                ],
                markdown="# Demo",
            )
            calls = {"post": 0, "put": 0, "poll": 0, "download": 0}

            def handler(request: httpx.Request) -> httpx.Response:
                if request.method == "POST" and request.url.path == "/api/v4/file-urls/batch":
                    calls["post"] += 1
                    payload = json.loads(request.content.decode("utf-8"))
                    self.assertEqual(payload["files"][0]["name"], "report.pdf")
                    self.assertEqual(payload["model_version"], "vlm")
                    return httpx.Response(
                        200,
                        json={
                            "code": 0,
                            "data": {"batch_id": batch_id, "file_urls": [upload_url]},
                            "msg": "ok",
                        },
                    )
                if request.method == "PUT" and str(request.url) == upload_url:
                    calls["put"] += 1
                    self.assertTrue(request.content.startswith(b"%PDF-1.4"))
                    return httpx.Response(200)
                if request.method == "GET" and request.url.path == f"/api/v4/extract-results/batch/{batch_id}":
                    calls["poll"] += 1
                    return httpx.Response(
                        200,
                        json={
                            "code": 0,
                            "data": {
                                "batch_id": batch_id,
                                "extract_result": [{"state": "done", "full_zip_url": zip_url, "err_msg": ""}],
                            },
                            "msg": "ok",
                        },
                    )
                if request.method == "GET" and str(request.url) == zip_url:
                    calls["download"] += 1
                    return httpx.Response(200, content=zip_bytes)
                raise AssertionError(f"Unexpected request: {request.method} {request.url}")

            parser = MineruPdfParser(
                artifact_root=artifact_root,
                api_token="test-token",
                client=httpx.Client(transport=httpx.MockTransport(handler)),
            )

            document = parser.parse(pdf_path)

            self.assertEqual(calls, {"post": 1, "put": 1, "poll": 1, "download": 1})
            self.assertEqual(document.markdown, "# Demo")
            self.assertEqual([item["kind"] for item in document.elements], ["heading", "paragraph", "table"])
            self.assertEqual(document.elements[0]["level"], 1)
            self.assertEqual(document.elements[-1]["table_column_count"], 2)
            self.assertTrue((artifact_root / document.doc_id / "result.zip").exists())
            self.assertTrue((artifact_root / document.doc_id / "content_list_v2.json").exists())

    def test_parser_reuses_cache_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "report.pdf"
            _write_pdf(pdf_path)
            parser = MineruPdfParser(artifact_root=root / "artifacts")
            _write_cached_artifacts(
                parser,
                pdf_path,
                pages=[
                    [
                        {
                            "type": "title",
                            "content": {"title_content": [{"type": "text", "content": "第二节 公司简介"}]},
                            "bbox": [1, 1, 2, 2],
                        },
                        {
                            "type": "paragraph",
                            "content": {
                                "paragraph_content": [{"type": "text", "content": "公司主营业务稳定。"}]
                            },
                            "bbox": [2, 2, 3, 3],
                        },
                    ]
                ],
            )
            parser._client = MagicMock()

            document = parser.parse(pdf_path)

            self.assertEqual(len(document.elements), 2)
            self.assertEqual(document.elements[0]["level"], 1)
            parser._client.post.assert_not_called()

    def test_force_parse_bypasses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "report.pdf"
            _write_pdf(pdf_path)
            batch_id = "batch-2"
            upload_url = "https://upload.example/report.pdf"
            zip_url = "https://cdn.example/result.zip"
            new_zip_bytes = _build_zip_bytes(
                pages=[
                    [
                        {
                            "type": "title",
                            "content": {"title_content": [{"type": "text", "content": "第三节 重要事项"}]},
                            "bbox": [1, 1, 2, 2],
                        }
                    ]
                ]
            )
            parser = MineruPdfParser(
                artifact_root=root / "artifacts",
                api_token="token",
                force_parse=True,
                client=httpx.Client(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            200,
                            json={"code": 0, "data": {"batch_id": batch_id, "file_urls": [upload_url]}, "msg": "ok"},
                        )
                        if request.method == "POST"
                        else httpx.Response(200)
                        if request.method == "PUT"
                        else httpx.Response(
                            200,
                            json={
                                "code": 0,
                                "data": {
                                    "batch_id": batch_id,
                                    "extract_result": [{"state": "done", "full_zip_url": zip_url}],
                                },
                                "msg": "ok",
                            },
                        )
                        if request.method == "GET" and request.url.path.startswith("/api/v4/extract-results/")
                        else httpx.Response(200, content=new_zip_bytes)
                    )
                ),
            )
            _write_cached_artifacts(
                parser,
                pdf_path,
                pages=[
                    [
                        {
                            "type": "title",
                            "content": {"title_content": [{"type": "text", "content": "旧缓存"}]},
                            "bbox": [1, 1, 2, 2],
                        }
                    ]
                ],
            )

            document = parser.parse(pdf_path)

            self.assertEqual(document.elements[0]["text"], "第三节 重要事项")

    def test_parser_raises_when_token_missing_and_cache_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "report.pdf"
            _write_pdf(pdf_path)
            parser = MineruPdfParser(artifact_root=root / "artifacts", api_token="")

            with self.assertRaisesRegex(ValueError, "MINERU_API_TOKEN"):
                parser.parse(pdf_path)

    def test_parser_falls_back_to_legacy_content_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "report.pdf"
            _write_pdf(pdf_path)
            parser = MineruPdfParser(artifact_root=root / "artifacts")
            _write_cached_artifacts(
                parser,
                pdf_path,
                legacy_items=[
                    {"type": "title", "text": "第一节 释义", "bbox": [1, 1, 2, 2], "page_idx": 0},
                    {
                        "type": "table",
                        "html": "<table><tr><td>项目</td><td>值</td></tr><tr><td>营业收入</td><td>100</td></tr></table>",
                        "bbox": [2, 2, 3, 3],
                        "page_idx": 0,
                    },
                ],
            )
            (parser.artifact_root / build_doc_id(pdf_path.resolve()) / "content_list_v2.json").unlink(missing_ok=True)

            document = parser.parse(pdf_path)

            self.assertEqual([item["kind"] for item in document.elements], ["heading", "table"])
            self.assertIn("营业收入", document.elements[1]["text"])


class MineruChunkerTests(unittest.TestCase):
    def test_chunker_groups_paragraphs_and_preserves_embedding_text(self) -> None:
        document = ParsedDocument(
            doc_id="doc-a",
            doc_name="doc-a.pdf",
            source_path="/tmp/doc-a.pdf",
            raw_doc={"parser": "mineru"},
            elements=[
                {"element_id": "h1", "kind": "heading", "text": "第一节 管理层讨论与分析", "level": 1, "page_start": 1, "page_end": 1, "provenance": [{"page": 1}]},
                {"element_id": "p1", "kind": "paragraph", "text": "第一段。", "page_start": 1, "page_end": 1, "provenance": [{"page": 1}]},
                {"element_id": "p2", "kind": "paragraph", "text": "第二段。", "page_start": 1, "page_end": 1, "provenance": [{"page": 1}]},
            ],
        )

        chunks = StructuredMineruChunker(max_chars=200).chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_path"], ["第一节 管理层讨论与分析"])
        self.assertEqual(chunks[0]["text"], "第一段。\n第二段。")
        self.assertIn("第一节 管理层讨论与分析", chunks[0]["embedding_text"])

    def test_chunker_splits_long_text_without_breaking_tables(self) -> None:
        document = ParsedDocument(
            doc_id="doc-b",
            doc_name="doc-b.pdf",
            source_path="/tmp/doc-b.pdf",
            raw_doc={"parser": "mineru"},
            elements=[
                {"element_id": "h1", "kind": "heading", "text": "第二节 财务摘要", "level": 1, "page_start": 1, "page_end": 1, "provenance": [{"page": 1}]},
                {"element_id": "p1", "kind": "paragraph", "text": "第一段。" * 80, "page_start": 1, "page_end": 1, "provenance": [{"page": 1}]},
            ],
        )

        chunks = StructuredMineruChunker(max_chars=180).chunk(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["chunk_type"] == "paragraph" for chunk in chunks))
        self.assertTrue(all(chunk["embedding_text"].startswith("第二节 财务摘要") for chunk in chunks))

    def test_chunker_keeps_table_in_single_chunk_and_preserves_html(self) -> None:
        document = ParsedDocument(
            doc_id="doc-c",
            doc_name="doc-c.pdf",
            source_path="/tmp/doc-c.pdf",
            raw_doc={"parser": "mineru"},
            elements=[
                {"element_id": "h1", "kind": "heading", "text": "第一节 主要会计数据", "level": 1, "page_start": 2, "page_end": 2, "provenance": [{"page": 2}]},
                {
                    "element_id": "t1",
                    "kind": "table",
                    "text": "主要会计数据\n项目 | 金额\n营业收入 | 100",
                    "table_html": "<table><tr><td>项目</td><td>金额</td></tr><tr><td>营业收入</td><td>100</td></tr></table>",
                    "table_caption": ["主要会计数据"],
                    "table_footnote": [],
                    "table_rows": [["项目", "金额"], ["营业收入", "100"]],
                    "table_row_html": ["<tr><td>项目</td><td>金额</td></tr>", "<tr><td>营业收入</td><td>100</td></tr>"],
                    "table_column_count": 2,
                    "table_continuation_hint": False,
                    "page_start": 2,
                    "page_end": 2,
                    "provenance": [{"page": 2}],
                },
            ],
        )

        chunks = StructuredMineruChunker(max_chars=10).chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_type"], "table")
        self.assertIn("营业收入", chunks[0]["text"])
        self.assertIn("<table>", chunks[0]["table_html"])
        self.assertEqual(chunks[0]["page_start"], 2)
        self.assertEqual(chunks[0]["page_end"], 2)

    def test_chunker_merges_cross_page_continuation_tables(self) -> None:
        document = ParsedDocument(
            doc_id="doc-d",
            doc_name="doc-d.pdf",
            source_path="/tmp/doc-d.pdf",
            raw_doc={"parser": "mineru"},
            elements=[
                {"element_id": "h1", "kind": "heading", "text": "第十节 财务报告", "level": 1, "page_start": 10, "page_end": 10, "provenance": [{"page": 10}]},
                {
                    "element_id": "t1",
                    "kind": "table",
                    "text": "项目 | 2024\n货币资金 | 100",
                    "table_html": "<table><tr><td>项目</td><td>2024</td></tr><tr><td>货币资金</td><td>100</td></tr></table>",
                    "table_caption": ["合并资产负债表"],
                    "table_footnote": [],
                    "table_rows": [["项目", "2024"], ["货币资金", "100"]],
                    "table_row_html": ["<tr><td>项目</td><td>2024</td></tr>", "<tr><td>货币资金</td><td>100</td></tr>"],
                    "table_column_count": 2,
                    "table_continuation_hint": False,
                    "page_start": 10,
                    "page_end": 10,
                    "provenance": [{"page": 10}],
                },
                {
                    "element_id": "t2",
                    "kind": "table",
                    "text": "项目 | 2024\n存货 | 200",
                    "table_html": "<table><tr><td>项目</td><td>2024</td></tr><tr><td>存货</td><td>200</td></tr></table>",
                    "table_caption": ["合并资产负债表（续表）"],
                    "table_footnote": [],
                    "table_rows": [["项目", "2024"], ["存货", "200"]],
                    "table_row_html": ["<tr><td>项目</td><td>2024</td></tr>", "<tr><td>存货</td><td>200</td></tr>"],
                    "table_column_count": 2,
                    "table_continuation_hint": True,
                    "page_start": 11,
                    "page_end": 11,
                    "provenance": [{"page": 11}],
                },
            ],
        )

        chunks = StructuredMineruChunker(max_chars=200).chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_start"], 10)
        self.assertEqual(chunks[0]["page_end"], 11)
        self.assertIn("货币资金", chunks[0]["text"])
        self.assertIn("存货", chunks[0]["text"])
        self.assertEqual(chunks[0]["table_html"].count("<tr>"), 3)

    def test_chunker_does_not_merge_unrelated_adjacent_tables(self) -> None:
        document = ParsedDocument(
            doc_id="doc-e",
            doc_name="doc-e.pdf",
            source_path="/tmp/doc-e.pdf",
            raw_doc={"parser": "mineru"},
            elements=[
                {
                    "element_id": "t1",
                    "kind": "table",
                    "text": "项目 | 金额\n营业收入 | 100",
                    "table_html": "<table><tr><td>项目</td><td>金额</td></tr><tr><td>营业收入</td><td>100</td></tr></table>",
                    "table_caption": [],
                    "table_footnote": [],
                    "table_rows": [["项目", "金额"], ["营业收入", "100"]],
                    "table_row_html": ["<tr><td>项目</td><td>金额</td></tr>", "<tr><td>营业收入</td><td>100</td></tr>"],
                    "table_column_count": 2,
                    "table_continuation_hint": False,
                    "page_start": 5,
                    "page_end": 5,
                    "provenance": [{"page": 5}],
                },
                {
                    "element_id": "t2",
                    "kind": "table",
                    "text": "地区 | 数量\n华东 | 20",
                    "table_html": "<table><tr><td>地区</td><td>数量</td></tr><tr><td>华东</td><td>20</td></tr></table>",
                    "table_caption": [],
                    "table_footnote": [],
                    "table_rows": [["地区", "数量"], ["华东", "20"]],
                    "table_row_html": ["<tr><td>地区</td><td>数量</td></tr>", "<tr><td>华东</td><td>20</td></tr>"],
                    "table_column_count": 2,
                    "table_continuation_hint": False,
                    "page_start": 6,
                    "page_end": 6,
                    "provenance": [{"page": 6}],
                },
            ],
        )

        chunks = StructuredMineruChunker(max_chars=200).chunk(document)

        self.assertEqual(len(chunks), 2)


class IngestionServiceTests(unittest.TestCase):
    def test_discover_pdf_files_reads_all_pdfs_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            first = raw_dir / "a.pdf"
            second = raw_dir / "b.pdf"
            _write_pdf(first)
            _write_pdf(second)

            discovered = discover_pdf_files(raw_dir)

        self.assertEqual(discovered, [first, second])

    def test_ingest_pdfs_writes_chunk_output_from_cached_mineru_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "report.pdf"
            _write_pdf(pdf_path)
            output_path = root / "chunks.json"
            artifact_root = root / "mineru"
            parser = MineruPdfParser(artifact_root=artifact_root)
            _write_cached_artifacts(
                parser,
                pdf_path,
                pages=[
                    [
                        {
                            "type": "title",
                            "content": {"title_content": [{"type": "text", "content": "第一节 经营情况"}]},
                            "bbox": [1, 1, 2, 2],
                        },
                        {
                            "type": "paragraph",
                            "content": {
                                "paragraph_content": [{"type": "text", "content": "公司整体经营稳定。"}]
                            },
                            "bbox": [2, 2, 3, 3],
                        },
                        {
                            "type": "table",
                            "content": {
                                "html": "<table><tr><td>项目</td><td>值</td></tr><tr><td>营业收入</td><td>100</td></tr></table>",
                                "table_caption": [{"type": "text", "content": "主要指标"}],
                                "table_footnote": [],
                            },
                            "bbox": [3, 3, 4, 4],
                        },
                    ]
                ],
            )

            chunks = ingest_pdfs([pdf_path], output_path, artifact_dir=artifact_root)

            saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved_chunks, chunks)
            self.assertEqual(len(saved_chunks), 2)
            self.assertEqual(saved_chunks[0]["chunk_type"], "paragraph")
            self.assertEqual(saved_chunks[1]["chunk_type"], "table")
            self.assertIn("table_html", saved_chunks[1])

    def test_ingestion_service_writes_chunks(self) -> None:
        parsed_document = ParsedDocument(
            doc_id="doc-f",
            doc_name="doc-f.pdf",
            source_path="/tmp/doc-f.pdf",
            raw_doc={"parser": "mineru"},
            elements=[
                {
                    "element_id": "h1",
                    "kind": "heading",
                    "text": "第一节",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "provenance": [{"page": 1}],
                },
                {
                    "element_id": "p1",
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
        service = IngestionService(parser=parser, chunk_strategy=StructuredMineruChunker(max_chars=200))

        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_path = Path(temp_dir) / "chunks.json"
            artifacts = IngestionArtifacts(chunks_path=chunks_path)

            chunks = service.ingest_pdfs([Path("/tmp/doc-f.pdf")], artifacts)

            saved_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        self.assertEqual(chunks, saved_chunks)
        self.assertEqual(saved_chunks[0]["section_path"], ["第一节"])


if __name__ == "__main__":
    unittest.main()
