from typing import Optional

from app.retrieval import RetrieverPort
from app.tables import JsonTableRepository
from app.tools.base import RegisteredTool, ToolRegistry, ToolSpec


def build_search_reports_tool(retriever: RetrieverPort) -> RegisteredTool:
    def handler(query: str, top_k: int = 3, doc_id: Optional[str] = None) -> dict:
        filters = {"doc_id": doc_id} if doc_id else None
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
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def build_search_tables_tool(table_repository: JsonTableRepository) -> RegisteredTool:
    def handler(
        doc_id: Optional[str] = None,
        query: Optional[str] = None,
        statement_type: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        resolved_doc_id = (doc_id or "").strip()
        resolved_query = (query or "").strip() or None
        resolved_statement_type = (statement_type or "").strip() or None
        if not resolved_doc_id:
            return {
                "doc_id": "",
                "tables": [],
                "error": "doc_id is required",
            }
        if resolved_query is None and resolved_statement_type is None:
            return {
                "doc_id": resolved_doc_id,
                "tables": [],
                "error": "query or statement_type is required",
            }
        tables = table_repository.search_tables(
            doc_id=resolved_doc_id,
            query=resolved_query,
            statement_type=resolved_statement_type,
            top_k=top_k,
        )
        return {
            "doc_id": resolved_doc_id,
            "tables": tables,
        }

    return RegisteredTool(
        spec=ToolSpec(
            name="search_tables",
            description=(
                "Search candidate tables inside a single indexed financial report. "
                "Use this first for statement and metric questions before falling back to text search."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document id to search within."},
                    "query": {
                        "type": ["string", "null"],
                        "description": "Optional natural-language query or metric name to match against table titles and contents.",
                    },
                    "statement_type": {
                        "type": ["string", "null"],
                        "enum": ["income_statement", "balance_sheet", "cash_flow", "key_metrics", None],
                        "description": "Optional statement type filter.",
                    },
                    "top_k": {"type": "integer", "description": "How many candidate tables to return.", "default": 5},
                },
                "required": ["doc_id"],
                "anyOf": [{"required": ["query"]}, {"required": ["statement_type"]}],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def build_extract_table_tool(table_repository: JsonTableRepository) -> RegisteredTool:
    def handler(doc_id: Optional[str] = None, table_id: Optional[str] = None) -> dict:
        resolved_doc_id = (doc_id or "").strip()
        resolved_table_id = (table_id or "").strip()
        if not resolved_doc_id:
            return {
                "doc_id": "",
                "table": None,
                "error": "doc_id is required",
            }
        if not resolved_table_id:
            return {
                "doc_id": resolved_doc_id,
                "table": None,
                "error": "table_id is required",
            }
        return {
            "doc_id": resolved_doc_id,
            "table": table_repository.get_table(doc_id=resolved_doc_id, table_id=resolved_table_id),
        }

    return RegisteredTool(
        spec=ToolSpec(
            name="extract_table",
            description="Return the full raw matrix for a specific table in an indexed financial report.",
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "Document id that owns the table."},
                    "table_id": {"type": "string", "description": "Stable table id returned by search_tables."},
                },
                "required": ["doc_id", "table_id"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def build_list_reports_tool(retriever: RetrieverPort) -> RegisteredTool:
    def handler() -> dict:
        documents = retriever.list_documents()
        return {"documents": [{"doc_id": item.doc_id, "doc_name": item.doc_name} for item in documents]}

    return RegisteredTool(
        spec=ToolSpec(
            name="list_reports",
            description="List the indexed annual reports that are currently available for question answering.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def build_default_tool_registry(
    retriever: RetrieverPort,
    table_repository: JsonTableRepository | None = None,
) -> ToolRegistry:
    repository = table_repository or JsonTableRepository.from_env()
    return ToolRegistry(
        [
            build_search_tables_tool(repository),
            build_extract_table_tool(repository),
            build_search_reports_tool(retriever),
            build_list_reports_tool(retriever),
        ]
    )
