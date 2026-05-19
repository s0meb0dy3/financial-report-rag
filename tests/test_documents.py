import io
import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from app.documents import DocumentManager
from app.domain import DocumentRef
from app.ingestion import ParsedDocument, build_doc_id
from app.session import SQLiteSessionStore


def _write_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


def _write_blank_pdf_bytes(page_count: int) -> bytes:
    payload = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    writer.write(payload)
    return payload.getvalue()


def _paragraph_element(text: str, page: int = 1) -> dict:
    return {
        "kind": "paragraph",
        "text": text,
        "page_start": page,
        "page_end": page,
        "provenance": [{"page": page}],
    }


class FakeParser:
    def parse(self, pdf_path: Path) -> ParsedDocument:
        path = Path(pdf_path).resolve()
        doc_id = build_doc_id(path)
        return ParsedDocument(
            doc_id=doc_id,
            doc_name=path.name,
            source_path=str(path),
            raw_doc={},
            elements=[_paragraph_element("营业总收入 100 亿元。")],
        )

    def close(self) -> None:
        pass


class FakeLargePdfParser(FakeParser):
    def parse(self, pdf_path: Path) -> ParsedDocument:
        page_count = len(PdfReader(str(pdf_path)).pages)
        if page_count <= 200:
            raise AssertionError("expected a large PDF upload")
        return super().parse(pdf_path)


class FakeRetriever:
    def __init__(self) -> None:
        self.documents: dict[str, str] = {}
        self.embedded_chunks: list[dict] = []
        self.deleted: list[str] = []
        self.invalidated = False

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        return [{**chunk, "embedding": [1.0, 0.0]} for chunk in chunks]

    def upsert_embedded_chunks(self, embedded_chunks: list[dict]) -> None:
        self.embedded_chunks.extend(embedded_chunks)
        for chunk in embedded_chunks:
            self.documents[chunk["doc_id"]] = chunk["doc_name"]

    def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.documents.pop(doc_id, None)

    def list_documents(self) -> list[DocumentRef]:
        return [
            DocumentRef(doc_id=doc_id, doc_name=doc_name)
            for doc_id, doc_name in sorted(self.documents.items())
        ]

    def invalidate_cache(self) -> None:
        self.invalidated = True


class DocumentManagerTests(unittest.TestCase):
    def _manager(self, root: Path, retriever: FakeRetriever | None = None) -> DocumentManager:
        return DocumentManager(
            retriever=retriever or FakeRetriever(),
            upload_dir=root / "raw" / "uploads",
            chunks_path=root / "processed" / "chunks.json",
            artifact_dir=root / "processed" / "mineru",
            parser_factory=FakeParser,
        )

    def test_upload_job_merges_chunks_without_overwriting_existing_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "processed" / "chunks.json"
            chunks_path.parent.mkdir(parents=True)
            chunks_path.write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "doc-a-chunk-1",
                            "doc_id": "doc-a",
                            "doc_name": "doc-a.pdf",
                            "source_path": "/tmp/doc-a.pdf",
                            "page": 1,
                            "section_path": [],
                            "chunk_type": "paragraph",
                            "text": "旧文档",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            retriever = FakeRetriever()
            manager = self._manager(root, retriever=retriever)

            job = manager.create_upload_job("report.pdf", _write_pdf_bytes())
            manager.run_job(job.job_id)

            saved = json.loads(chunks_path.read_text(encoding="utf-8"))

        self.assertEqual({chunk["doc_id"] for chunk in saved}, {"doc-a", job.doc_id})
        self.assertEqual(manager.get_job(job.job_id).status, "succeeded")
        self.assertTrue(retriever.embedded_chunks)

    def test_upload_large_pdf_still_indexes_one_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = FakeRetriever()
            manager = DocumentManager(
                retriever=retriever,
                upload_dir=root / "raw" / "uploads",
                chunks_path=root / "processed" / "chunks.json",
                artifact_dir=root / "processed" / "mineru",
                parser_factory=FakeLargePdfParser,
            )

            job = manager.create_upload_job("large-report.pdf", _write_blank_pdf_bytes(201))
            manager.run_job(job.job_id)

            documents = manager.list_documents()

        self.assertEqual(manager.get_job(job.job_id).status, "succeeded")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].doc_id, job.doc_id)

    def test_delete_document_removes_runtime_outputs_and_clears_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chunks_path = root / "processed" / "chunks.json"
            chunks_path.parent.mkdir(parents=True)
            chunks_path.write_text(
                json.dumps(
                    [
                        {"chunk_id": "a-1", "doc_id": "doc-a", "doc_name": "doc-a.pdf", "text": "A"},
                        {"chunk_id": "b-1", "doc_id": "doc-b", "doc_name": "doc-b.pdf", "text": "B"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "processed" / "mineru" / "doc-a"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "content_list_v2.json").write_text("[]", encoding="utf-8")
            store = SQLiteSessionStore(root / "sessions.sqlite3")
            session = store.create_session("s1", title="会话", doc_ids=["doc-a", "doc-b"])
            retriever = FakeRetriever()
            retriever.documents = {"doc-a": "doc-a.pdf", "doc-b": "doc-b.pdf"}
            manager = DocumentManager(
                retriever=retriever,
                session_store=store,
                upload_dir=root / "raw" / "uploads",
                chunks_path=chunks_path,
                artifact_dir=root / "processed" / "mineru",
                parser_factory=FakeParser,
            )

            deleted = manager.delete_document("doc-a")
            saved = json.loads(chunks_path.read_text(encoding="utf-8"))
            updated_session = store.get_session(session.id)

        self.assertTrue(deleted)
        self.assertEqual([chunk["doc_id"] for chunk in saved], ["doc-b"])
        self.assertFalse(artifact_dir.exists())
        self.assertIn("doc-a", retriever.deleted)
        self.assertEqual(updated_session.doc_id, "doc-b")
        self.assertEqual(updated_session.doc_ids, ["doc-b"])


if __name__ == "__main__":
    unittest.main()
