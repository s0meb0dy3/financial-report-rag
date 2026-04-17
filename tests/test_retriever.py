import os
import unittest
from unittest.mock import MagicMock, patch

from app.retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    Retriever,
)


class RetrieverTests(unittest.TestCase):
    def test_from_env_reads_defaults_and_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            retriever = Retriever.from_env()

        self.assertEqual(retriever.api_key, "test-key")
        self.assertEqual(retriever.base_url, DEFAULT_OPENROUTER_BASE_URL)
        self.assertEqual(retriever.embedding_model, DEFAULT_EMBEDDING_MODEL)

    def test_from_env_reads_environment_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_BASE_URL": "https://example.com/api/v1",
                "EMBEDDING_MODEL": "custom-model",
            },
            clear=True,
        ):
            retriever = Retriever.from_env()

        self.assertEqual(retriever.api_key, "test-key")
        self.assertEqual(retriever.base_url, "https://example.com/api/v1")
        self.assertEqual(retriever.embedding_model, "custom-model")

    def test_extract_embeddings_reads_openrouter_shape(self) -> None:
        response_json = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2]},
                {"object": "embedding", "embedding": [0.3, 0.4]},
            ],
        }
        retriever = Retriever(api_key="test-key")

        embeddings = retriever._extract_embeddings(response_json)

        self.assertEqual(embeddings, [[0.1, 0.2], [0.3, 0.4]])

    def test_parse_embedding_response_tolerates_leading_whitespace(self) -> None:
        retriever = Retriever(api_key="test-key")

        parsed = retriever._parse_embedding_response(
            MagicMock(text="\n   {\"data\": [{\"embedding\": [0.1, 0.2]}]}")
        )

        self.assertEqual(parsed["data"][0]["embedding"], [0.1, 0.2])

    def test_index_chunks_batches_embedding_requests(self) -> None:
        vector_store = MagicMock()
        retriever = Retriever(api_key="test-key", vector_store=vector_store, batch_size=2)
        chunks = [
            {
                "chunk_id": f"doc-a-page-1-chunk-{index}",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 1,
                "text": f"chunk-{index}",
            }
            for index in range(1, 6)
        ]

        with patch.object(
            retriever,
            "embed",
            side_effect=[
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.5], [0.1, 0.9]],
                [[0.2, 0.8]],
            ],
        ) as mock_embed:
            embedded = retriever.index_chunks(chunks)

        self.assertEqual(mock_embed.call_count, 3)
        self.assertEqual(mock_embed.call_args_list[0].args[0], ["chunk-1", "chunk-2"])
        self.assertEqual(mock_embed.call_args_list[1].args[0], ["chunk-3", "chunk-4"])
        self.assertEqual(mock_embed.call_args_list[2].args[0], ["chunk-5"])
        self.assertEqual(len(embedded), 5)
        vector_store.upsert_documents.assert_called_once()

    def test_index_chunks_embeds_text_and_upserts_to_vector_store(self) -> None:
        vector_store = MagicMock()
        retriever = Retriever(api_key="test-key", vector_store=vector_store)
        chunks = [
            {
                "chunk_id": "doc-a-page-1-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 1,
                "text": "收入增长",
            }
        ]

        with patch.object(retriever, "embed", return_value=[[1.0, 0.0]]) as mock_embed:
            embedded = retriever.index_chunks(chunks)

        mock_embed.assert_called_once_with(["收入增长"])
        vector_store.upsert_documents.assert_called_once()
        self.assertEqual(embedded[0]["embedding"], [1.0, 0.0])

    def test_index_chunks_condenses_long_table_for_embedding_only(self) -> None:
        vector_store = MagicMock()
        retriever = Retriever(api_key="test-key", vector_store=vector_store)
        long_text = "项目 | 金额\n" + ("数据行\n" * 3000)
        chunks = [
            {
                "chunk_id": "doc-a-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page_start": 61,
                "page_end": 63,
                "chunk_type": "table",
                "section_path": ["母公司资产负债表"],
                "text": long_text,
            }
        ]

        with patch.object(retriever, "embed", return_value=[[1.0, 0.0]]) as mock_embed:
            embedded = retriever.index_chunks(chunks)

        embedding_input = mock_embed.call_args.args[0][0]
        self.assertLess(len(embedding_input), len(long_text))
        self.assertIn("table chunk", embedding_input)
        self.assertEqual(embedded[0]["text"], long_text)
        vector_store.upsert_documents.assert_called_once()

    def test_search_embeds_query_and_delegates_to_vector_store(self) -> None:
        vector_store = MagicMock()
        vector_store.search.return_value = [
            {
                "chunk_id": "doc-a-page-1-chunk-1",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 1,
                "text": "收入增长",
                "score": 0.91,
            }
        ]
        retriever = Retriever(api_key="test-key", vector_store=vector_store)

        with patch.object(retriever, "embed", return_value=[[1.0, 0.0]]) as mock_embed:
            results = retriever.search("营业总收入是多少？", top_k=2, filters={"doc_id": "doc-a"})

        mock_embed.assert_called_once_with(["营业总收入是多少？"])
        vector_store.search.assert_called_once_with([1.0, 0.0], top_k=2, filters={"doc_id": "doc-a"})
        self.assertEqual(results[0]["doc_name"], "doc-a.pdf")

    def test_list_documents_delegates_to_vector_store(self) -> None:
        vector_store = MagicMock()
        vector_store.list_documents.return_value = [
            {"doc_id": "doc-a", "doc_name": "doc-a.pdf"},
            {"doc_id": "doc-b", "doc_name": "doc-b.pdf"},
        ]
        retriever = Retriever(api_key="test-key", vector_store=vector_store)

        results = retriever.list_documents()

        vector_store.list_documents.assert_called_once_with()
        self.assertEqual(
            results,
            [
                {"doc_id": "doc-a", "doc_name": "doc-a.pdf"},
                {"doc_id": "doc-b", "doc_name": "doc-b.pdf"},
            ],
        )

    def test_close_shuts_down_existing_client(self) -> None:
        retriever = Retriever(api_key="test-key")
        client = MagicMock()
        retriever._client = client

        retriever.close()

        client.close.assert_called_once()
        self.assertIsNone(retriever._client)

    def test_close_also_closes_owned_vector_store(self) -> None:
        vector_store = MagicMock()
        retriever = Retriever(api_key="test-key", vector_store=vector_store)

        retriever.close()

        vector_store.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
