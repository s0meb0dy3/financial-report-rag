from dataclasses import asdict, dataclass, field
from typing import Any

from app.domain import Evidence
from app.retrieval import RetrieverPort
from app.tables import JsonTableRepository


@dataclass(frozen=True)
class RagCitation:
    doc_id: str
    doc_name: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagEvidence:
    doc_id: str
    doc_name: str
    page: int | None
    text: str
    score: float
    chunk_id: str = ""
    chunk_type: str = ""
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None

    @classmethod
    def from_evidence(cls, item: Evidence) -> "RagEvidence":
        return cls(
            doc_id=item.doc_id,
            doc_name=item.doc_name,
            page=item.page,
            text=item.text,
            score=item.score,
            chunk_id=item.chunk_id,
            chunk_type=item.chunk_type,
            section_path=list(item.section_path),
            page_start=item.page_start,
            page_end=item.page_end,
        )

    def citation(self) -> RagCitation:
        return RagCitation(doc_id=self.doc_id, doc_name=self.doc_name, page=self.page)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagTable:
    table_id: str
    doc_id: str
    doc_name: str
    title: str
    page_start: int | None = None
    page_end: int | None = None
    statement_type_guess: str | None = None
    preview_matrix: list[list[str]] = field(default_factory=list)
    score: float = 0.0

    @classmethod
    def from_payload(cls, item: dict[str, Any]) -> "RagTable":
        return cls(
            table_id=str(item.get("table_id", "")),
            doc_id=str(item.get("doc_id", "")),
            doc_name=str(item.get("doc_name", "")),
            title=str(item.get("title", "")),
            page_start=item.get("page_start") if isinstance(item.get("page_start"), int) else None,
            page_end=item.get("page_end") if isinstance(item.get("page_end"), int) else None,
            statement_type_guess=item.get("statement_type_guess"),
            preview_matrix=[
                [str(cell) for cell in row]
                for row in item.get("preview_matrix", [])
                if isinstance(row, list)
            ],
            score=float(item.get("score", 0.0) or 0.0),
        )

    def citation(self) -> RagCitation:
        return RagCitation(doc_id=self.doc_id, doc_name=self.doc_name, page=self.page_start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RagResult:
    query: str
    retrieval_queries: list[str]
    evidences: list[RagEvidence]
    tables: list[RagTable]
    citations: list[RagCitation]
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_context(self) -> bool:
        return bool(self.evidences or self.tables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_queries": self.retrieval_queries,
            "evidences": [item.to_dict() for item in self.evidences],
            "tables": [item.to_dict() for item in self.tables],
            "citations": [item.to_dict() for item in self.citations],
            "metadata": self.metadata,
        }


class RagService:
    """Stable boundary for financial-report retrieval used by chat and external tools."""

    def __init__(
        self,
        retriever: RetrieverPort,
        table_repository: JsonTableRepository | None = None,
        *,
        default_top_k: int = 5,
        table_top_k: int = 3,
    ):
        self.retriever = retriever
        self.table_repository = table_repository or JsonTableRepository.from_env()
        self.default_top_k = default_top_k
        self.table_top_k = table_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        include_tables: bool = True,
    ) -> RagResult:
        resolved_query = query.strip()
        if not resolved_query:
            raise ValueError("query must not be blank")

        selected_doc_ids = _normalize_doc_ids(doc_ids, fallback_doc_id=doc_id)
        filters = _doc_filter(selected_doc_ids)
        resolved_top_k = max(1, top_k or self.default_top_k)
        metadata: dict[str, Any] = {"doc_ids": selected_doc_ids}

        try:
            raw_evidences = self.retriever.search(
                resolved_query,
                top_k=resolved_top_k,
                filters=filters,
            )
        except Exception as exc:
            raw_evidences = []
            metadata["retrieval_error"] = str(exc)

        evidences = [RagEvidence.from_evidence(item) for item in raw_evidences]
        tables: list[RagTable] = []
        if include_tables:
            try:
                tables = [
                    RagTable.from_payload(item)
                    for item in self.table_repository.search_tables(
                        query=resolved_query,
                        top_k=min(resolved_top_k, self.table_top_k),
                        doc_ids=selected_doc_ids or None,
                    )
                ]
            except Exception as exc:
                metadata["table_error"] = str(exc)

        retrieval_queries_getter = getattr(self.retriever, "get_last_retrieval_queries", None)
        retrieval_queries = (
            retrieval_queries_getter() if callable(retrieval_queries_getter) else [resolved_query]
        )
        if not isinstance(retrieval_queries, list) or not retrieval_queries:
            retrieval_queries = [resolved_query]

        return RagResult(
            query=resolved_query,
            retrieval_queries=[str(item) for item in retrieval_queries],
            evidences=evidences,
            tables=tables,
            citations=_dedupe_citations([*(item.citation() for item in evidences), *(item.citation() for item in tables)]),
            metadata=metadata,
        )

    def list_documents(self):
        return self.retriever.list_documents()

    def close(self) -> None:
        close = getattr(self.retriever, "close", None)
        if callable(close):
            close()


def _normalize_doc_ids(
    doc_ids: list[str] | None = None,
    *,
    fallback_doc_id: str | None = None,
) -> list[str]:
    candidates = doc_ids if doc_ids is not None else ([fallback_doc_id] if fallback_doc_id else [])
    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            continue
        doc_id = value.strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        normalized.append(doc_id)
    return normalized


def _doc_filter(doc_ids: list[str]) -> dict[str, Any] | None:
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"doc_id": {"$in": doc_ids}}


def _dedupe_citations(citations: list[RagCitation]) -> list[RagCitation]:
    seen: set[tuple[str, int | None]] = set()
    output: list[RagCitation] = []
    for item in citations:
        if not item.doc_id:
            continue
        key = (item.doc_id, item.page)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output
