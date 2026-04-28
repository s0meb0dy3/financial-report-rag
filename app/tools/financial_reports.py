from typing import Optional

from app.retrieval import RetrieverPort
from app.tables import JsonTableRepository
from app.tools.base import RegisteredTool, ToolRegistry, ToolSpec
from app.tools.charts import build_create_chart_tool


def build_search_reports_tool(retriever: RetrieverPort) -> RegisteredTool:
    def handler(
        query: str,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        doc_ids: list[str] | None = None,
    ) -> dict:
        selected_doc_ids = _normalize_doc_ids(doc_ids, fallback_doc_id=doc_id)
        filters = _doc_filter(selected_doc_ids)
        results = retriever.search(query, top_k=top_k, filters=filters)
        retrieval_queries_getter = getattr(retriever, "get_last_retrieval_queries", None)
        retrieval_queries = retrieval_queries_getter() if callable(retrieval_queries_getter) else None
        if not isinstance(retrieval_queries, list) or not all(
            isinstance(item, str) for item in retrieval_queries
        ):
            retrieval_queries = [query]
        return {
            "query": query,
            "retrieval_queries": retrieval_queries or [query],
            "results": [
                {
                    "doc_id": result.doc_id,
                    "doc_name": result.doc_name,
                    "page": result.page,
                    "page_start": result.page_start,
                    "page_end": result.page_end,
                    "chunk_type": result.chunk_type,
                    "section_path": result.section_path,
                    "text": result.text,
                    "score": result.score,
                }
                for result in results
            ],
        }

    return RegisteredTool(
        spec=ToolSpec(
            name="search_reports",
            description="Search indexed financial report chunks and return supporting evidence with document names and page numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to run against the indexed reports."},
                    "top_k": {"type": "integer", "description": "How many results to return.", "default": 3},
                    "doc_id": {"type": ["string", "null"], "description": "Optional document id filter."},
                    "doc_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional document id filters. Use this for multi-document analysis.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def build_search_tables_tool(table_repository: JsonTableRepository) -> RegisteredTool:
    def handler(
        query: str = "",
        top_k: int = 5,
        doc_id: Optional[str] = None,
        doc_ids: list[str] | None = None,
        statement_type: str | None = None,
    ) -> dict:
        selected_doc_ids = _normalize_doc_ids(doc_ids, fallback_doc_id=doc_id)
        tables = table_repository.search_tables(
            query=query,
            top_k=top_k,
            doc_ids=selected_doc_ids or None,
            statement_type=statement_type,
        )
        return {
            "query": query,
            "doc_ids": selected_doc_ids,
            "statement_type": statement_type,
            "tables": tables,
        }

    return RegisteredTool(
        spec=ToolSpec(
            name="search_tables",
            description=(
                "Search structured financial table chunks and return candidate tables with previews, "
                "page ranges, document names, and table ids. Use this first for financial metrics, "
                "statement tables, multi-year figures, and chart data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Metric, caption, or table query."},
                    "top_k": {"type": "integer", "description": "How many candidate tables to return.", "default": 5},
                    "doc_id": {"type": ["string", "null"], "description": "Optional document id filter."},
                    "doc_ids": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional document id filters. Use this for multi-document analysis.",
                    },
                    "statement_type": {
                        "type": ["string", "null"],
                        "enum": ["income_statement", "balance_sheet", "cash_flow", "key_metrics", None],
                        "description": "Optional financial statement/table type hint.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def build_get_table_tool(table_repository: JsonTableRepository) -> RegisteredTool:
    def handler(table_id: str, doc_id: Optional[str] = None) -> dict:
        table = table_repository.get_table(table_id=table_id, doc_id=doc_id)
        return {
            "table_id": table_id,
            "doc_id": doc_id,
            "table": table,
            "error": None if table is not None else "table not found",
        }

    return RegisteredTool(
        spec=ToolSpec(
            name="get_table",
            description=(
                "Read a full structured table by table_id after search_tables returns a candidate. "
                "Use this when the table preview is insufficient to answer or chart accurately."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "table_id": {"type": "string", "description": "Table id returned by search_tables."},
                    "doc_id": {"type": ["string", "null"], "description": "Optional document id filter."},
                },
                "required": ["table_id"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


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


def _doc_filter(doc_ids: list[str]) -> dict | None:
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}
    return {"doc_id": {"$in": doc_ids}}


def build_default_tool_registry(
    retriever: RetrieverPort,
    table_repository: JsonTableRepository | None = None,
) -> ToolRegistry:
    resolved_table_repository = table_repository or JsonTableRepository.from_env()
    return ToolRegistry(
        [
            build_search_reports_tool(retriever),
            build_search_tables_tool(resolved_table_repository),
            build_get_table_tool(resolved_table_repository),
            build_create_chart_tool(),
        ]
    )
