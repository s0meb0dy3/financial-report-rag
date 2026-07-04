import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.documents import DocumentService, DocumentServiceError


def make_pdf_bytes() -> bytes:
    import pymupdf

    pdf = pymupdf.open()
    try:
        pdf.new_page()
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

    def test_lists_uploaded_pdf_as_unparsed_document(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            service = DocumentService(raw_dir=root / "raw", mineru_dir=root / "mineru")

            doc = service.save_upload("demo.pdf", make_pdf_bytes())
            docs = service.list_documents()

        self.assertEqual(doc.parsed, False)
        self.assertEqual(docs[0].id, doc.id)
        self.assertEqual(docs[0].name, "demo.pdf")
        self.assertEqual(docs[0].parsed, False)

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
