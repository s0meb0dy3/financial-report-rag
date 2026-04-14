from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    tool_name: str
    output: dict[str, Any]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[..., dict[str, Any]]

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, output=self.handler(**kwargs))


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool]):
        self._tools = {tool.spec.name: tool for tool in tools}

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.spec.to_openai_definition() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        return self._tools[tool_name].execute(**kwargs).output
