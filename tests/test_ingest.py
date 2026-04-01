import json
import tempfile
import unittest
from pathlib import Path

from ingest import chunk_page_text, ingest_pdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "茅台24年年度报告.pdf"


class IngestPdfTests(unittest.TestCase):
    def test_chunk_page_text_splits_long_text_into_multiple_chunks(self) -> None:
        text = "第一段。" * 120 + "\n" + "第二段。" * 120

        chunks = chunk_page_text(text, page_number=1, max_chars=300, overlap_chars=60)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["page"], 1)
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))

    def test_ingest_pdf_writes_chunked_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chunks.json"

            chunks = ingest_pdf(PDF_PATH, output_path)

            self.assertGreaterEqual(len(chunks), 7)
            self.assertTrue(output_path.exists())

            saved_chunks = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(saved_chunks, chunks)
            self.assertEqual(saved_chunks[0]["page"], 1)
            self.assertIn("贵州茅台酒股份有限公司", saved_chunks[0]["text"])
            self.assertTrue(all(chunk["chunk_id"] for chunk in saved_chunks))
            self.assertTrue(all(chunk["text"].strip() for chunk in saved_chunks))


if __name__ == "__main__":
    unittest.main()
