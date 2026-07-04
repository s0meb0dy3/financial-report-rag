from typing import Any
import re

from app.documents import DocumentService, DocumentServiceError


class ListReportsTool:
    """Expose the local parsed financial reports to the model."""

    name = "list_reports"

    def __init__(self, document_service: DocumentService):
        self.document_service = document_service

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "List locally available financial report PDFs and their document ids.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "reports": [
                {
                    "doc_id": doc.id,
                    "doc_name": doc.name,
                    "page_count": doc.page_count,
                }
                for doc in self.document_service.list_documents()
            ]
        }


class ReadTableOfContentsTool:
    """Read the table of contents (bookmarks) from a financial report PDF."""

    name = "read_toc"

    def __init__(self, document_service: DocumentService):
        self.document_service = document_service

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read the table of contents from a local financial report PDF. "
                    "Returns chapter/section titles with page numbers. "
                    "Use this before reading specific pages so you know which page to look at. "
                    "The returned page numbers are physical page numbers — pass them directly to read_pdf_page."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document id from list_reports or a citation.",
                        },
                    },
                    "required": ["doc_id"],
                    "additionalProperties": False,
                },
            },
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(arguments.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError("doc_id must not be blank")
        try:
            return self.document_service.get_toc(doc_id)
        except DocumentServiceError as exc:
            raise ValueError(str(exc)) from exc


class ReadPdfPageTool:
    """Read one precise page from a MinerU parsed PDF."""

    name = "read_pdf_page"

    def __init__(self, document_service: DocumentService, *, default_max_chars: int = 12000):
        self.document_service = document_service
        self.default_max_chars = default_max_chars

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read a specific page from a local financial report PDF using the MinerU parsed result. "
                    "Use this when the user asks about a specific report page or asks you to verify source text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document id from list_reports or a citation.",
                        },
                        "page": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "1-based PDF page number.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1000,
                            "maximum": 20000,
                            "description": "Maximum characters of page text to return.",
                        },
                    },
                    "required": ["doc_id", "page"],
                    "additionalProperties": False,
                },
            },
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(arguments.get("doc_id") or "").strip()
        page = _as_int(arguments.get("page"), default=0)
        if not doc_id:
            raise ValueError("doc_id must not be blank")
        if page < 1:
            raise ValueError("page must be a positive integer")

        try:
            parsed_page = self.document_service.read_page(doc_id, page)
        except DocumentServiceError as exc:
            raise ValueError(str(exc)) from exc

        max_chars = _as_int(arguments.get("max_chars"), default=self.default_max_chars)
        max_chars = min(20000, max(1000, max_chars))
        text = parsed_page.text[:max_chars]
        truncated = len(parsed_page.text) > len(text)
        citation = {
            "doc_id": parsed_page.doc_id,
            "doc_name": parsed_page.doc_name,
            "page": parsed_page.page,
        }
        return {
            "doc_id": parsed_page.doc_id,
            "doc_name": parsed_page.doc_name,
            "page": parsed_page.page,
            "text": text,
            "truncated": truncated,
            "blocks": _trim_blocks(parsed_page.blocks, max_chars),
            "citations": [citation],
        }


class SearchReportTextTool:
    """Plain text search over MinerU parsed report pages."""

    name = "search_report_text"

    def __init__(self, document_service: DocumentService, *, default_max_results: int = 5):
        self.document_service = document_service
        self.default_max_results = default_max_results

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search local parsed financial report text and return matching pages. "
                    "Use this before read_pdf_page when you need to locate where a topic appears."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Text to search for in parsed report pages.",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "Optional document id from list_reports. If omitted, search all reports.",
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Maximum number of page matches to return.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query must not be blank")

        doc_id = str(arguments.get("doc_id") or "").strip()
        max_results = min(20, max(1, _as_int(arguments.get("max_results"), default=self.default_max_results)))
        try:
            docs = [self.document_service.get_document(doc_id)] if doc_id else self.document_service.list_documents()
        except DocumentServiceError as exc:
            raise ValueError(str(exc)) from exc
        query_lower = query.lower()
        terms = _query_terms(query)

        results: list[dict[str, Any]] = []
        for doc in docs:
            for page in range(1, doc.page_count + 1):
                try:
                    parsed_page = self.document_service.read_page(doc.id, page)
                except DocumentServiceError:
                    continue
                match = _match_page(parsed_page.text, query_lower, terms)
                if match is None:
                    continue
                results.append({
                    "doc_id": parsed_page.doc_id,
                    "doc_name": parsed_page.doc_name,
                    "page": parsed_page.page,
                    "snippet": _snippet(parsed_page.text, match["index"], match["length"]),
                    "score": match["score"],
                    "matched_terms": match["terms"],
                })
        results.sort(key=lambda item: (-int(item["score"]), str(item["doc_name"]), int(item["page"])))
        results = results[:max_results]
        citations = [
            {
                "doc_id": item["doc_id"],
                "doc_name": item["doc_name"],
                "page": item["page"],
            }
            for item in results
        ]
        return {"query": query, "terms": terms, "results": results, "citations": citations}


def _trim_blocks(blocks: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    remaining = max_chars
    for block in blocks:
        if remaining <= 0:
            break
        text = str(block.get("text") or "")
        if not text:
            continue
        trimmed = text[:remaining]
        item = dict(block)
        item["text"] = trimmed
        if len(text) > len(trimmed):
            item["truncated"] = True
        result.append(item)
        remaining -= len(trimmed) + 1
    return result


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _query_terms(query: str) -> list[str]:
    terms = [item.lower() for item in re.findall(r"[\w一-鿿]+", query) if item.strip()]
    if query.lower() not in terms:
        terms.insert(0, query.lower())
    return list(dict.fromkeys(term for term in terms if term))


def _match_page(text: str, query_lower: str, terms: list[str]) -> dict[str, Any] | None:
    text_lower = text.lower()
    exact_index = text_lower.find(query_lower)
    if exact_index >= 0:
        return {
            "index": exact_index,
            "length": len(query_lower),
            "terms": [query_lower],
            "score": 1000 + text_lower.count(query_lower) * 10,
        }

    matched: list[str] = []
    first_index: int | None = None
    score = 0
    for term in terms:
        if term == query_lower:
            continue
        index = text_lower.find(term)
        if index < 0:
            continue
        matched.append(term)
        first_index = index if first_index is None else min(first_index, index)
        score += 100 + min(20, text_lower.count(term) * 2) + min(20, len(term))
    if not matched or first_index is None:
        return None
    return {"index": first_index, "length": len(matched[0]), "terms": matched, "score": score}


def _snippet(text: str, index: int, query_length: int, *, radius: int = 80) -> str:
    start = max(0, index - radius)
    end = min(len(text), index + query_length + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
