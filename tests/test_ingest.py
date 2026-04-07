import json
import tempfile
import unittest
from shutil import copy2
from pathlib import Path

from ingest import chunk_page_text, discover_pdf_files, ingest_pdfs


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "茅台2024年年度报告完整版.pdf"
SECOND_PDF_PATH = PROJECT_ROOT / "长江电力2024年报.pdf"


class IngestPdfTests(unittest.TestCase):
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
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chunks.json"

            chunks = ingest_pdfs([PDF_PATH, SECOND_PDF_PATH], output_path)

            self.assertGreaterEqual(len(chunks), 7)
            self.assertTrue(output_path.exists())

            saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved_chunks, chunks)
            self.assertEqual(saved_chunks[0]["page"], 1)
            self.assertTrue(all(chunk["chunk_id"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["text"].strip() for chunk in saved_chunks))
            self.assertTrue(all(chunk["doc_id"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["doc_name"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["source_path"] for chunk in saved_chunks))
            self.assertEqual({chunk["doc_name"] for chunk in saved_chunks}, {PDF_PATH.name, SECOND_PDF_PATH.name})
            self.assertTrue(
                all(Path(chunk["source_path"]).name == chunk["doc_name"] for chunk in saved_chunks)
            )


if __name__ == "__main__":
    unittest.main()
