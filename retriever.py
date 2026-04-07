import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

from vector_store import ChromaVectorStore, VectorStore


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Retriever:
    """负责 embedding 生成和向量库检索"""

    @classmethod
    def from_env(cls) -> "Retriever":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: float = 120,
        vector_store: Optional[VectorStore] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self.vector_store = vector_store or ChromaVectorStore.from_env()
        self._owns_vector_store = vector_store is None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_embeddings(response_json: dict[str, Any]) -> list[list[float]]:
        data = response_json.get("data", [])
        return [item["embedding"] for item in data if "embedding" in item]

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        return json.loads(path.read_text(encoding="utf-8"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={"model": self.embedding_model, "input": texts},
        )
        response.raise_for_status()
        embeddings = self._extract_embeddings(response.json())
        if not embeddings:
            raise ValueError("No embedding data received")
        return embeddings

    def index_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        embeddings = self.embed([chunk["text"] for chunk in chunks])
        embedded_chunks = [
            {**chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.vector_store.upsert_documents(embedded_chunks)
        return embedded_chunks

    def index_chunks_from_path(self, chunks_path: Path) -> list[dict[str, Any]]:
        chunks = self._load_json(chunks_path)
        return self.index_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embed([query])[0]
        return self.vector_store.search(query_embedding, top_k=top_k, filters=filters)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        if self.vector_store:
            self.vector_store.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build embeddings and index chunks into ChromaDB.")
    parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks.json",
        help="Path to the chunk JSON file",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    chunks_path = project_root / args.chunks_path

    with Retriever.from_env() as retriever:
        embedded = retriever.index_chunks_from_path(chunks_path)
    print(f"Indexed {len(embedded)} chunks from {chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
