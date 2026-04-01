import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv


load_dotenv()

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Retriever:
    """基于向量相似度的文本块检索器"""

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
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

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
    def _write_json(data: list[dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量嵌入"""
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

    @staticmethod
    def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(v * v for v in vector_a))
        norm_b = math.sqrt(sum(v * v for v in vector_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def load_embeddings(self, path: Path) -> list[dict[str, Any]]:
        """从文件加载已嵌入的文本块"""
        return self._load_json(path)

    def save_embeddings(self, embedded_chunks: list[dict[str, Any]], path: Path) -> None:
        """保存嵌入结果到文件"""
        self._write_json(embedded_chunks, path)

    def build_embeddings(self, chunks_path: Path, output_path: Path) -> list[dict[str, Any]]:
        """从原始 chunks 文件构建嵌入并保存"""
        chunks = self._load_json(chunks_path)
        embeddings = self.embed([chunk["text"] for chunk in chunks])
        embedded_chunks = [
            {**chunk, "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.save_embeddings(embedded_chunks, output_path)
        return embedded_chunks

    def rank_by_similarity(
        self,
        query_embedding: list[float],
        embedded_chunks: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """按与查询向量的相似度排序文本块"""
        scored = [
            {**chunk, "score": self.cosine_similarity(query_embedding, chunk["embedding"])}
            for chunk in embedded_chunks
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search(
        self,
        query: str,
        embeddings_path: Path,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """加载嵌入并检索与查询最相关的文本块"""
        embedded_chunks = self.load_embeddings(embeddings_path)
        query_embedding = self.embed([query])[0]
        return self.rank_by_similarity(query_embedding, embedded_chunks, top_k=top_k)

    def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    chunks_path = project_root / "data" / "processed" / "chunks.json"
    embeddings_path = project_root / "data" / "processed" / "embeddings.json"

    with Retriever.from_env() as retriever:
        retriever.build_embeddings(chunks_path, embeddings_path)
    print(f"Wrote embeddings to {embeddings_path}")
