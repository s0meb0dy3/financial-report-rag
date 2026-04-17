from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias


ParsedElement: TypeAlias = dict[str, Any]
ChunkRecord: TypeAlias = dict[str, Any]
TableRecord: TypeAlias = dict[str, Any]


@dataclass(slots=True)
class ParsedDocument:
    doc_id: str
    doc_name: str
    source_path: str
    raw_doc: dict[str, Any]
    elements: list[ParsedElement]
    markdown: str | None = None
    page_map: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionArtifacts:
    chunks_path: Path
    docling_json_dir: Path
    markdown_dir: Path
    tables_path: Path | None = None
    export_markdown: bool = True


class DocumentParser(Protocol):
    def parse(self, pdf_path: Path) -> ParsedDocument:
        ...


class ChunkStrategy(Protocol):
    def chunk(self, document: ParsedDocument) -> list[ChunkRecord]:
        ...
