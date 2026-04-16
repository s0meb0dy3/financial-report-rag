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

    def get_all_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
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
                self._build_metadata(chunk)
                for chunk in chunks
            ],
            embeddings=[chunk["embedding"] for chunk in chunks],
        )

    @staticmethod
    def _build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "doc_id": chunk["doc_id"],
            "doc_name": chunk["doc_name"],
            "source_path": chunk["source_path"],
            "chunk_type": chunk.get("chunk_type", ""),
            "section_path_text": " > ".join(chunk.get("section_path", [])),
        }
        for key in ("page", "page_start", "page_end"):
            value = chunk.get(key)
            if value is not None:
                metadata[key] = value
        return metadata

    @staticmethod
    def _parse_section_path(metadata: dict[str, Any]) -> list[str]:
        raw_value = str(metadata.get("section_path_text", "")).strip()
        if not raw_value:
            return []
        return [part.strip() for part in raw_value.split(" > ") if part.strip()]

    @classmethod
    def _build_evidence(
        cls,
        *,
        chunk_id: str,
        document: str,
        metadata: dict[str, Any] | None,
        score: float = 0.0,
    ) -> Evidence:
        metadata = metadata or {}
        return Evidence(
            doc_id=metadata.get("doc_id", ""),
            doc_name=metadata.get("doc_name", ""),
            page=metadata.get("page"),
            text=document,
            score=score,
            chunk_id=chunk_id,
            source_path=metadata.get("source_path", ""),
            chunk_type=metadata.get("chunk_type", ""),
            section_path=cls._parse_section_path(metadata),
            page_start=metadata.get("page_start"),
            page_end=metadata.get("page_end"),
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
            results.append(
                self._build_evidence(
                    chunk_id=chunk_id,
                    document=document or "",
                    metadata=metadata,
                    score=1 - float(distance),
                )
            )
        return results

    def get_all_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Evidence]:
        response = self._collection.get(
            where=filters or None,
            include=["documents", "metadatas"],
        )
        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        return [
            self._build_evidence(
                chunk_id=chunk_id,
                document=document or "",
                metadata=metadata,
            )
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        ]

    def close(self) -> None:
        self._collection = None
        self._client = None
