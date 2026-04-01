import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from retriever import DEFAULT_EMBEDDING_MODEL, DEFAULT_OPENROUTER_BASE_URL, Retriever


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

    def test_cosine_similarity_prefers_same_direction(self) -> None:
        self.assertGreater(Retriever.cosine_similarity([1.0, 0.0], [1.0, 0.0]), 0.99)
        self.assertLess(Retriever.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.01)

    def test_rank_chunks_by_similarity_returns_top_match_first(self) -> None:
        chunks = [
            {"chunk_id": "a", "page": 1, "text": "收入增长", "embedding": [1.0, 0.0]},
            {"chunk_id": "b", "page": 2, "text": "董事会信息", "embedding": [0.0, 1.0]},
        ]
        retriever = Retriever(api_key="test-key")

        results = retriever.rank_by_similarity([1.0, 0.0], chunks, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "a")
        self.assertIn("score", results[0])

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

    def test_save_and_load_embeddings_round_trip(self) -> None:
        retriever = Retriever(api_key="test-key")
        embedded_chunks = [{"chunk_id": "a", "page": 1, "text": "收入增长", "embedding": [1.0, 0.0]}]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "embeddings.json"
            retriever.save_embeddings(embedded_chunks, path)

            loaded = retriever.load_embeddings(path)

        self.assertEqual(loaded, embedded_chunks)

    def test_close_shuts_down_existing_client(self) -> None:
        retriever = Retriever(api_key="test-key")
        client = MagicMock()
        retriever._client = client

        retriever.close()

        client.close.assert_called_once()
        self.assertIsNone(retriever._client)


if __name__ == "__main__":
    unittest.main()
