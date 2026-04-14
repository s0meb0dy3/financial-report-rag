from typing import Optional

from app.retrieval import RetrieverPort
from app.tools.base import RegisteredTool, ToolRegistry, ToolSpec


def build_search_reports_tool(retriever: RetrieverPort) -> RegisteredTool:
    def handler(query: str, top_k: int = 3, doc_id: Optional[str] = None) -> dict:
        filters = {"doc_id": doc_id} if doc_id else None
        results = retriever.search(query, top_k=top_k, filters=filters)
        return {
            "query": query,
            "results": [
                {
                    "doc_id": result.doc_id,
                    "doc_name": result.doc_name,
                    "page": result.page,
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


def build_default_tool_registry(retriever: RetrieverPort) -> ToolRegistry:
    return ToolRegistry([build_search_reports_tool(retriever), build_list_reports_tool(retriever)])
