from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionResult:
    id: str
    name: str
    arguments: dict[str, Any]
    status: str
    content: dict[str, Any]
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "content": self.content,
            "citations": self.citations,
            "error": self.error,
        }


class ChatTool(Protocol):
    name: str

    def schema(self) -> dict[str, Any]:
        raise NotImplementedError

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
