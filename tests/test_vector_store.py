import tempfile
import unittest
from pathlib import Path

from app.retrieval import ChromaVectorStore


class ChromaVectorStoreTests(unittest.TestCase):
    def test_upsert_and_search_persist_across_instances(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a-page-1-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "table",
                "section_path": ["第一章", "财务摘要"],
                "text": "营业总收入 1,741.44 亿元。",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "doc-b-page-1-chunk-1",
                "doc_id": "doc-b",
                "doc_name": "doc-b.pdf",
                "source_path": "/tmp/doc-b.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "paragraph",
                "section_path": ["第二章", "利润情况"],
                "text": "净利润 862.28 亿元。",
                "embedding": [0.0, 1.0],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            persist_dir = Path(temp_dir) / "chroma"

            store = ChromaVectorStore(persist_dir=persist_dir, collection_name="test")
            store.upsert_documents(chunks)
            store.close()

            reopened = ChromaVectorStore(persist_dir=persist_dir, collection_name="test")
            results = reopened.search([1.0, 0.0], top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "doc-a-page-1-chunk-1")
        self.assertEqual(results[0].doc_name, "doc-a.pdf")
        self.assertEqual(results[0].page, 1)
        self.assertEqual(results[0].page_start, 1)
        self.assertEqual(results[0].page_end, 1)
        self.assertEqual(results[0].chunk_type, "table")
        self.assertEqual(results[0].section_path, ["第一章", "财务摘要"])
        self.assertIsInstance(results[0].score, float)

    def test_search_supports_metadata_filters(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a-page-1-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "paragraph",
                "section_path": ["第一章"],
                "text": "营业总收入 1,741.44 亿元。",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "doc-b-page-1-chunk-1",
                "doc_id": "doc-b",
                "doc_name": "doc-b.pdf",
                "source_path": "/tmp/doc-b.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
                "chunk_type": "paragraph",
                "section_path": ["第二章"],
                "text": "营业总收入 99 亿元。",
                "embedding": [1.0, 0.0],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                persist_dir=Path(temp_dir) / "chroma",
                collection_name="test",
            )
            store.upsert_documents(chunks)

            results = store.search([1.0, 0.0], top_k=3, filters={"doc_id": "doc-b"})
            multi_results = store.search(
                [1.0, 0.0],
                top_k=3,
                filters={"doc_id": {"$in": ["doc-a", "doc-b"]}},
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "doc-b")
        self.assertEqual({item.doc_id for item in multi_results}, {"doc-a", "doc-b"})

    def test_delete_document_removes_indexed_vectors(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a-page-1-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 1,
                "chunk_type": "paragraph",
                "section_path": ["第一章"],
                "text": "营业总收入 1,741.44 亿元。",
                "embedding": [1.0, 0.0],
            },
            {
                "chunk_id": "doc-b-page-1-chunk-1",
                "doc_id": "doc-b",
                "doc_name": "doc-b.pdf",
                "source_path": "/tmp/doc-b.pdf",
                "page": 1,
                "chunk_type": "paragraph",
                "section_path": ["第二章"],
                "text": "净利润 862.28 亿元。",
                "embedding": [0.0, 1.0],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                persist_dir=Path(temp_dir) / "chroma",
                collection_name="test",
            )
            store.upsert_documents(chunks)

            store.delete_document("doc-a")
            documents = store.list_documents()
            deleted_results = store.search([1.0, 0.0], top_k=3, filters={"doc_id": "doc-a"})

        self.assertEqual([document.doc_id for document in documents], ["doc-b"])
        self.assertEqual(deleted_results, [])

    def test_get_all_documents_returns_rich_metadata(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a-page-2-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 2,
                "page_start": 2,
                "page_end": 3,
                "chunk_type": "table",
                "section_path": ["第一章", "主要会计数据"],
                "text": "经营活动产生的现金流量净额。",
                "embedding": [1.0, 0.0],
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                persist_dir=Path(temp_dir) / "chroma",
                collection_name="test",
            )
            store.upsert_documents(chunks)

            results = store.get_all_documents()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_type, "table")
        self.assertEqual(results[0].section_path, ["第一章", "主要会计数据"])
        self.assertEqual(results[0].page_start, 2)
        self.assertEqual(results[0].page_end, 3)


if __name__ == "__main__":
    unittest.main()
