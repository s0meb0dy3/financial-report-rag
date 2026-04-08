import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from vector_store import ChromaVectorStore, VectorStore


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_QUERY_REWRITE_MODEL = "qwen/qwen3.6-plus:free"


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


class QueryRewriter:
    """将原始问题改写为更适合检索的多个查询"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        chat_model: Optional[str] = None,
        max_rewrites: int = 2,
        client: Optional[OpenAI] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = base_url
        self.chat_model = chat_model or os.environ.get("CHAT_MODEL", DEFAULT_QUERY_REWRITE_MODEL)
        self.max_rewrites = max_rewrites
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _parse_queries(content: str) -> list[str]:
        if not content:
            return []
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []

        if isinstance(payload, dict):
            items = payload.get("queries", [])
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        return [item.strip() for item in items if isinstance(item, str) and item.strip()]

    @staticmethod
    def _format_history(history_messages: Optional[list[dict[str, str]]]) -> str:
        if not history_messages:
            return "无"
        return "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history_messages
            if message.get("content")
        )

    def rewrite(
        self,
        question: str,
        history_messages: Optional[list[dict[str, str]]] = None,
    ) -> list[str]:
        queries = [question.strip()]
        if not queries[0]:
            return []

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是财报检索查询改写器。"
                            "请基于用户问题和最近对话历史，生成最多 2 条适合向量检索的查询改写。"
                            "要求补全公司名、年份、指标名和指代信息；不要回答问题；必须只返回一个 JSON 对象。"
                            '输出 schema 固定为 {"queries":["...","..."]}。'
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"当前问题：{question}\n"
                            f"最近对话：\n{self._format_history(history_messages)}\n"
                            "请按固定 schema 输出最多 2 条检索查询改写。"
                        ),
                    },
                ],
            )
            rewritten = self._parse_queries(self._extract_content(response))
        except Exception:
            rewritten = []

        seen = {queries[0]}
        for item in rewritten:
            if item in seen:
                continue
            queries.append(item)
            seen.add(item)
            if len(queries) >= self.max_rewrites + 1:
                break
        return queries

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None


class MultiQueryRetriever:
    """用多条改写查询做召回，再合并排序结果"""

    def __init__(
        self,
        base_retriever: Retriever,
        query_rewriter: QueryRewriter,
    ):
        self.base_retriever = base_retriever
        self.query_rewriter = query_rewriter

    def search_with_queries(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
        history_messages: Optional[list[dict[str, str]]] = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        queries = self.query_rewriter.rewrite(query, history_messages=history_messages)
        merged: dict[str, dict[str, Any]] = {}

        for rewritten_query in queries:
            results = self.base_retriever.search(rewritten_query, top_k=top_k, filters=filters)
            for result in results:
                chunk_id = result["chunk_id"]
                current = merged.get(chunk_id)
                if current is None or result.get("score", 0.0) > current.get("score", 0.0):
                    merged[chunk_id] = result

        ranked = sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)
        return ranked[:top_k], queries

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
        history_messages: Optional[list[dict[str, str]]] = None,
    ) -> list[dict[str, Any]]:
        results, _ = self.search_with_queries(
            query,
            top_k=top_k,
            filters=filters,
            history_messages=history_messages,
        )
        return results

    def close(self) -> None:
        self.query_rewriter.close()


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
