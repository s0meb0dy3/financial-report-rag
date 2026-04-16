from dataclasses import dataclass, field
from typing import Any

from app.messages import BaseMessage


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
class Citation:
    doc_id: str
    doc_name: str
    page: int | None


@dataclass
class DocumentRef:
    doc_id: str
    doc_name: str


@dataclass
class ToolTrace:
    tool_name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    tool_call_id: str = ""


@dataclass
class ConversationState:
    messages: list[BaseMessage] = field(default_factory=list)

    def __post_init__(self) -> None:
        for message in self.messages:
            if not isinstance(message, BaseMessage):
                raise TypeError("ConversationState.messages must contain BaseMessage instances")


@dataclass
class TurnResult:
    answer: str
    citations: list[Citation]
    tool_traces: list[ToolTrace]
    updated_state: ConversationState
