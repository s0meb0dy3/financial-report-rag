import json
import os
import unittest
from unittest.mock import MagicMock, patch

from retriever import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    MultiQueryRetriever,
    QueryRewriter,
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

    def test_query_rewriter_returns_original_and_rewritten_queries(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps(
            {
                "queries": [
                    "贵州茅台 2024 年 营业总收入",
                    "茅台年报 营业总收入",
                    "贵州茅台 2024 年 营业总收入",
                ]
            },
            ensure_ascii=False,
        )
        client.chat.completions.create.return_value = response

        rewriter = QueryRewriter(
            api_key="test-key",
            chat_model="test-chat-model",
            client=client,
        )

        queries = rewriter.rewrite(
            "它的营收是多少？",
            history_messages=[
                {"role": "user", "content": "贵州茅台 2024 年怎么样？"},
                {"role": "assistant", "content": "表现不错。"},
            ],
        )

        self.assertEqual(
            queries,
            [
                "它的营收是多少？",
                "贵州茅台 2024 年 营业总收入",
                "茅台年报 营业总收入",
            ],
        )
        client.chat.completions.create.assert_called_once()
        request_kwargs = client.chat.completions.create.call_args.kwargs
        request_messages = request_kwargs["messages"]
        self.assertIn("贵州茅台 2024 年怎么样？", request_messages[1]["content"])
        self.assertEqual(request_kwargs["response_format"], {"type": "json_object"})

    def test_query_rewriter_falls_back_to_original_query_on_invalid_response(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "not-json"
        client.chat.completions.create.return_value = response

        rewriter = QueryRewriter(
            api_key="test-key",
            chat_model="test-chat-model",
            client=client,
        )

        queries = rewriter.rewrite("营业总收入是多少？")

        self.assertEqual(queries, ["营业总收入是多少？"])

    def test_multi_query_retriever_merges_deduplicates_and_sorts_results(self) -> None:
        base_retriever = MagicMock()
        base_retriever.search.side_effect = [
            [
                {"chunk_id": "a", "doc_id": "doc-a", "doc_name": "a.pdf", "page": 1, "text": "收入", "score": 0.82},
                {"chunk_id": "b", "doc_id": "doc-b", "doc_name": "b.pdf", "page": 2, "text": "利润", "score": 0.76},
            ],
            [
                {"chunk_id": "a", "doc_id": "doc-a", "doc_name": "a.pdf", "page": 1, "text": "收入", "score": 0.91},
                {"chunk_id": "c", "doc_id": "doc-c", "doc_name": "c.pdf", "page": 3, "text": "现金流", "score": 0.88},
            ],
            [
                {"chunk_id": "d", "doc_id": "doc-d", "doc_name": "d.pdf", "page": 4, "text": "费用", "score": 0.7},
            ],
        ]
        query_rewriter = MagicMock()
        query_rewriter.rewrite.return_value = ["原始问题", "改写一", "改写二"]
        multi_retriever = MultiQueryRetriever(base_retriever=base_retriever, query_rewriter=query_rewriter)

        results = multi_retriever.search(
            "原始问题",
            top_k=3,
            filters={"doc_id": "doc-a"},
            history_messages=[{"role": "user", "content": "上一轮问题"}],
        )

        self.assertEqual([result["chunk_id"] for result in results], ["a", "c", "b"])
        self.assertEqual(results[0]["score"], 0.91)
        query_rewriter.rewrite.assert_called_once_with("原始问题", history_messages=[{"role": "user", "content": "上一轮问题"}])
        self.assertEqual(base_retriever.search.call_count, 3)
        for call in base_retriever.search.call_args_list:
            self.assertEqual(call.kwargs["filters"], {"doc_id": "doc-a"})
            self.assertEqual(call.kwargs["top_k"], 3)

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
