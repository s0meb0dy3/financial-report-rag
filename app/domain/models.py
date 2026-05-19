from dataclasses import dataclass, field


@dataclass
class Evidence:
    doc_id: str
    doc_name: str
    page: int | None
    text: str
    score: float = 0.0
    chunk_id: str = ""
    source_path: str = ""
    chunk_type: str = ""
    section_path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class DocumentRef:
    doc_id: str
    doc_name: str
