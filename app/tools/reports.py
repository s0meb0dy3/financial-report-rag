from typing import Any

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
            "blocks": parsed_page.blocks,
            "citations": [citation],
        }


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
