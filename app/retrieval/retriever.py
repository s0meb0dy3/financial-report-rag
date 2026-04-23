import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional, Protocol

import httpx

from app.domain import DocumentRef, Evidence
from app.retrieval.vector_store import ChromaVectorStore, VectorStore


DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_BATCH_SIZE = 200


class RetrieverPort(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:
        ...

    def list_documents(self) -> list[DocumentRef]:
        ...

    def get_last_retrieval_queries(self) -> list[str]:
        ...


class ChromaRetriever:
    @classmethod
    def from_env(cls) -> "ChromaRetriever":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)),
        )

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: float = 120,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        vector_store: Optional[VectorStore] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self._client: Optional[httpx.Client] = None
        self.vector_store = vector_store or ChromaVectorStore.from_env()
        self._last_retrieval_queries: list[str] = []

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

    @staticmethod
    def _parse_embedding_response(response: httpx.Response) -> dict[str, Any]:
        try:
            return json.loads(response.text.lstrip())
        except json.JSONDecodeError as exc:
            preview = response.text[:500]
            raise ValueError(
                "Embedding API returned invalid JSON. "
                f"status={response.status_code} content_type={response.headers.get('content-type')} "
                f"body_preview={preview!r}"
            ) from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=self._headers(),
            json={"model": self.embedding_model, "input": texts},
        )
        response.raise_for_status()
        embeddings = self._extract_embeddings(self._parse_embedding_response(response))
        if not embeddings:
            raise ValueError("No embedding data received")
        return embeddings

    def index_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        embedded_chunks = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embeddings = self.embed([chunk.get("embedding_text", chunk["text"]) for chunk in batch])
            embedded_chunks.extend(
                {**chunk, "embedding": embedding}
                for chunk, embedding in zip(batch, embeddings, strict=True)
            )
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
    ) -> list[Evidence]:
        self._last_retrieval_queries = [query]
        query_embedding = self.embed([query])[0]
        return self.vector_store.search(query_embedding, top_k=top_k, filters=filters)

    def get_all_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        return self.vector_store.get_all_documents(filters=filters)

    def list_documents(self) -> list[DocumentRef]:
        return self.vector_store.list_documents()

    def get_last_retrieval_queries(self) -> list[str]:
        return list(self._last_retrieval_queries)

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
        if self.vector_store:
            self.vector_store.close()

    def __enter__(self) -> "ChromaRetriever":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


Retriever = ChromaRetriever


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build embeddings and index chunks into ChromaDB.",
        add_help=add_help,
    )
    parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks.json",
        help="Path to the chunk JSON file",
    )
    return parser


def run_command(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    chunks_path = project_root / args.chunks_path

    with ChromaRetriever.from_env() as retriever:
        embedded = retriever.index_chunks_from_path(chunks_path)
    print(f"Indexed {len(embedded)} chunks from {chunks_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_command(args)
