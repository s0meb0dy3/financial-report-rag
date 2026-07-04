import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.documents import DocumentService, DocumentServiceError
import app.documents.service as document_service


def make_pdf_bytes(text: str = "") -> bytes:
    import pymupdf

    pdf = pymupdf.open()
    try:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
        return pdf.tobytes()
    finally:
        pdf.close()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DocumentServiceTests(unittest.TestCase):
    def test_lists_and_reads_single_mineru_document(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "raw" / "report.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-1.4")
            artifact = root / "mineru" / "doc-a"
            write_json(
                artifact / "manifest.json",
                {
                    "doc_id": "doc-a",
                    "file_name": "report.pdf",
                    "source_path": str(pdf),
                },
            )
            write_json(
                artifact / "content_list_v2.json",
                [
                    [
                        {
                            "type": "title",
                            "content": {"title_content": [{"type": "text", "content": "标题"}]},
                            "bbox": [1, 2, 3, 4],
                        },
                        {
                            "type": "paragraph",
                            "content": {"paragraph_content": [{"type": "text", "content": "正文"}]},
                        },
                    ]
                ],
            )

            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")
            docs = service.list_documents()
            page = service.read_page("doc-a", 1)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_count, 1)
        self.assertEqual(page.text, "标题\n正文")
        self.assertEqual(page.blocks[0]["bbox"], [1, 2, 3, 4])

    def test_reads_split_document_with_global_page_number(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "raw" / "split.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(b"%PDF-1.4")
            artifact = root / "mineru" / "split-doc"
            part = artifact / "parts" / "part-002"
            write_json(
                artifact / "manifest.json",
                {
                    "doc_id": "split-doc",
                    "file_name": "split.pdf",
                    "source_path": str(pdf),
                    "split": True,
                    "page_count": 3,
                    "parts": [
                        {"part_index": 1, "page_start": 1, "page_end": 2},
                        {
                            "part_index": 2,
                            "page_start": 3,
                            "page_end": 3,
                            "artifact_dir": str(part),
                        },
                    ],
                },
            )
            write_json(
                part / "content_list_v2.json",
                [
                    [
                        {
                            "type": "paragraph",
                            "content": {"paragraph_content": [{"type": "text", "content": "第三页"}]},
                        }
                    ]
                ],
            )

            page = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru").read_page("split-doc", 3)

        self.assertEqual(page.page, 3)
        self.assertEqual(page.text, "第三页")

    def test_upload_parses_pdf_text(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")

            doc = service.save_upload("demo.pdf", make_pdf_bytes("hello revenue"))
            docs = service.list_documents()
            page = service.read_page(doc.id, 1)

        self.assertEqual(doc.parsed, True)
        self.assertEqual(docs[0].id, doc.id)
        self.assertEqual(docs[0].name, "demo.pdf")
        self.assertEqual(docs[0].parsed, True)
        self.assertIn("hello revenue", page.text)

    def test_upload_uses_mineru_api_when_configured(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_parse(api_key, pdf_path, content, page_count, artifact_dir):
                self.assertEqual(api_key, "mineru-key")
                write_json(
                    artifact_dir / "manifest.json",
                    {
                        "doc_id": "upload-demo",
                        "file_name": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page_count": page_count,
                        "parser": "mineru_api_precise",
                    },
                )
                write_json(
                    artifact_dir / "content_list_v2.json",
                    [[{"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "MinerU 精准解析"}]}}]],
                )

            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru", mineru_api_key="mineru-key")
            with patch("app.documents.service._parse_with_mineru_api", side_effect=fake_parse) as parse:
                doc = service.save_upload("demo.pdf", make_pdf_bytes("ignored local text"))
                page = service.read_page(doc.id, 1)

        self.assertTrue(parse.called)
        self.assertEqual(doc.parsed, True)
        self.assertIn("MinerU 精准解析", page.text)

    def test_mineru_page_ranges_cap_at_200_pages(self) -> None:
        self.assertEqual(
            document_service._mineru_page_ranges(401),
            [(1, 200), (201, 400), (401, 401)],
        )

    def test_mineru_upload_url_accepts_api_string_shape(self) -> None:
        self.assertEqual(
            document_service._mineru_upload_url("https://example.com/upload"),
            "https://example.com/upload",
        )
        self.assertEqual(
            document_service._mineru_upload_url({"url": "https://example.com/upload"}),
            "https://example.com/upload",
        )

    def test_normalize_content_list_prefers_mineru_v2_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "abc_content_list.json", [{"text": "旧版", "page_idx": 0}])
            write_json(
                root / "abc_content_list_v2.json",
                [[{"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "新版"}]}}]],
            )

            document_service._normalize_content_list(root)
            payload = json.loads((root / "content_list_v2.json").read_text(encoding="utf-8"))

        self.assertEqual(payload[0][0]["content"]["paragraph_content"][0]["content"], "新版")

    def test_normalize_content_list_groups_legacy_output_by_page(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                root / "abc_content_list.json",
                [
                    {"type": "text", "text": "第一页", "page_idx": 0},
                    {"type": "text", "text": "第二页", "page_idx": 1},
                ],
            )

            document_service._normalize_content_list(root)
            payload = json.loads((root / "content_list_v2.json").read_text(encoding="utf-8"))

        self.assertEqual(payload[0][0]["text"], "第一页")
        self.assertEqual(payload[1][0]["text"], "第二页")

    def test_deletes_uploaded_document(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")
            doc = service.save_upload("demo.pdf", make_pdf_bytes())

            removed = service.delete_document(doc.id)
            docs = service.list_documents()

        self.assertTrue(removed)
        self.assertEqual(docs, [])

    def test_rejects_invalid_uploaded_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            service = DocumentService(raw_dir=Path(directory) / "raw", mineru_dir=Path(directory) / "mineru")

            with self.assertRaises(DocumentServiceError):
                service.save_upload("demo.pdf", b"not a pdf")

    def test_does_not_delete_parsed_builtin_document(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "raw" / "report.pdf"
            pdf.parent.mkdir(parents=True)
            pdf.write_bytes(make_pdf_bytes())
            artifact = root / "mineru" / "doc-a"
            write_json(artifact / "manifest.json", {"doc_id": "doc-a", "file_name": "report.pdf", "source_path": str(pdf)})
            write_json(artifact / "content_list_v2.json", [[{"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "正文"}]}}]])
            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")

            removed = service.delete_document("doc-a")
            pdf_exists = pdf.exists()

        self.assertFalse(removed)
        self.assertTrue(pdf_exists)


if __name__ == "__main__":
    unittest.main()
