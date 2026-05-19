import unittest

from app.domain import DocumentRef, Evidence
from app.rag import RagService


class FakeRetriever:
    def __init__(self) -> None:
        self.last_call = None

    def search(self, query, top_k=3, filters=None):
        self.last_call = {"query": query, "top_k": top_k, "filters": filters}
        return [
            Evidence(
                doc_id="doc-a",
                doc_name="doc-a.pdf",
                page=8,
                text="营业收入增长。",
                score=0.5,
                chunk_id="chunk-a",
                section_path=["主要会计数据"],
            )
        ]

    def get_last_retrieval_queries(self):
        return ["营业收入", "主要会计数据 营业收入"]

    def list_documents(self):
        return [DocumentRef(doc_id="doc-a", doc_name="doc-a.pdf")]


class FakeTableRepository:
    def __init__(self) -> None:
        self.last_call = None

    def search_tables(self, **kwargs):
        self.last_call = kwargs
        return [
            {
                "table_id": "table-a",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "title": "主要会计数据",
                "page_start": 8,
                "page_end": 8,
                "preview_matrix": [["指标", "2024"], ["营业收入", "100"]],
                "score": 2.5,
            }
        ]


class RagServiceTests(unittest.TestCase):
    def test_retrieve_returns_text_tables_and_deduped_citations(self) -> None:
        retriever = FakeRetriever()
        tables = FakeTableRepository()
        service = RagService(retriever=retriever, table_repository=tables)

        result = service.retrieve(" 营业收入 ", top_k=4, doc_ids=["doc-a", "doc-b"])

        self.assertEqual(retriever.last_call["query"], "营业收入")
        self.assertEqual(retriever.last_call["top_k"], 4)
        self.assertEqual(retriever.last_call["filters"], {"doc_id": {"$in": ["doc-a", "doc-b"]}})
        self.assertEqual(tables.last_call["doc_ids"], ["doc-a", "doc-b"])
        self.assertEqual(result.retrieval_queries[1], "主要会计数据 营业收入")
        self.assertEqual(result.evidences[0].text, "营业收入增长。")
        self.assertEqual(result.tables[0].table_id, "table-a")
        self.assertEqual(len(result.citations), 1)
        self.assertTrue(result.has_context())

    def test_retrieve_can_skip_tables(self) -> None:
        tables = FakeTableRepository()
        service = RagService(retriever=FakeRetriever(), table_repository=tables)

        result = service.retrieve("营业收入", include_tables=False)

        self.assertEqual(result.tables, [])
        self.assertIsNone(tables.last_call)

    def test_retrieve_surfaces_retrieval_error_as_empty_context(self) -> None:
        class BrokenRetriever(FakeRetriever):
            def search(self, query, top_k=3, filters=None):
                raise RuntimeError("index missing")

        service = RagService(retriever=BrokenRetriever(), table_repository=FakeTableRepository())

        result = service.retrieve("营业收入", include_tables=False)

        self.assertFalse(result.has_context())
        self.assertEqual(result.metadata["retrieval_error"], "index missing")


if __name__ == "__main__":
    unittest.main()
