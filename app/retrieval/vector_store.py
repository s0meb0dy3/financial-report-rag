import os
from pathlib import Path
from typing import Any, Optional, Protocol

import chromadb

from app.domain import DocumentRef, Evidence


DEFAULT_CHROMA_PERSIST_DIR = "data/chroma"
DEFAULT_CHROMA_COLLECTION_NAME = "financial-report-chunks"


class VectorStore(Protocol):
    def upsert_documents(self, chunks: list[dict[str, Any]]) -> None:
        ...

    def list_documents(self) -> list[DocumentRef]:
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:
        ...

    def close(self) -> None:
        ...


class ChromaVectorStore:
    @classmethod
    def from_env(cls) -> "ChromaVectorStore":
        project_root = Path(__file__).resolve().parents[2]
        persist_dir = Path(
            os.environ.get(
                "CHROMA_PERSIST_DIR",
                str(project_root / DEFAULT_CHROMA_PERSIST_DIR),
            )
        )
        return cls(
            persist_dir=persist_dir,
            collection_name=os.environ.get(
                "CHROMA_COLLECTION_NAME",
                DEFAULT_CHROMA_COLLECTION_NAME,
            ),
        )

    def __init__(self, persist_dir: Path, collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_documents(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk["doc_id"],
                    "doc_name": chunk["doc_name"],
                    "source_path": chunk["source_path"],
                    "page": chunk["page"],
                }
                for chunk in chunks
            ],
            embeddings=[chunk["embedding"] for chunk in chunks],
        )

    def list_documents(self) -> list[DocumentRef]:
        response = self._collection.get(include=["metadatas"])
        metadatas = response.get("metadatas", [])
        seen = set()
        documents = []
        for metadata in metadatas:
            metadata = metadata or {}
            key = (metadata.get("doc_id", ""), metadata.get("doc_name", ""))
            if key in seen:
                continue
            seen.add(key)
            documents.append(DocumentRef(doc_id=key[0], doc_name=key[1]))
        documents.sort(key=lambda item: item.doc_name)
        return documents

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:
        response = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            results.append(
                Evidence(
                    doc_id=metadata.get("doc_id", ""),
                    doc_name=metadata.get("doc_name", ""),
                    page=metadata.get("page"),
                    text=document,
                    score=1 - float(distance),
                    chunk_id=chunk_id,
                    source_path=metadata.get("source_path", ""),
                )
            )
        return results

    def close(self) -> None:
        self._collection = None
        self._client = None
