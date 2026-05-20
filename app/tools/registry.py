import json
import re
from typing import Any

from app.tools.types import ChatTool, ToolCall, ToolExecutionResult


class ToolRegistry:
    """Single boundary for tool schemas, aliases, provider quirks, and execution."""

    def __init__(self, tools: list[ChatTool] | None = None):
        self._tools = list(tools or [])
        self._by_name: dict[str, ChatTool] = {}
        for tool in self._tools:
            self._by_name[tool.name] = tool
            for alias in getattr(tool, "aliases", ()):
                if isinstance(alias, str) and alias:
                    self._by_name[alias] = tool

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools]

    def execute(self, call: ToolCall) -> ToolExecutionResult:
        tool = self._by_name.get(call.name)
        if tool is None:
            return ToolExecutionResult(
                id=call.id,
                name=call.name,
                arguments=call.arguments,
                status="error",
                content={"error": f"Unknown tool: {call.name}"},
                error=f"Unknown tool: {call.name}",
            )
        try:
            content = tool.run(call.arguments)
        except Exception as exc:
            return ToolExecutionResult(
                id=call.id,
                name=call.name,
                arguments=call.arguments,
                status="error",
                content={"error": str(exc)},
                error=str(exc),
            )

        citations = content.get("citations") if isinstance(content, dict) else None
        return ToolExecutionResult(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            status="done",
            content=content if isinstance(content, dict) else {"result": content},
            citations=[item for item in citations if isinstance(item, dict)] if isinstance(citations, list) else [],
        )


def extract_tool_calls(message: Any) -> list[ToolCall]:
    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls is None and isinstance(message, dict):
        raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list | tuple):
        return []

    calls: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        function = _tool_call_function(raw_call)
        name = _object_value(function, "name")
        if not isinstance(name, str) or not name:
            continue
        raw_arguments = _object_value(function, "arguments")
        calls.append(
            ToolCall(
                id=str(_object_value(raw_call, "id") or f"tool-call-{index}"),
                name=name,
                arguments=parse_tool_arguments(raw_arguments),
            )
        )
    return calls


def extract_tool_call_deltas(delta: Any) -> list[dict[str, Any]]:
    raw_calls = getattr(delta, "tool_calls", None)
    if raw_calls is None and isinstance(delta, dict):
        raw_calls = delta.get("tool_calls")
    if not isinstance(raw_calls, list | tuple):
        return []

    result: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        function = _tool_call_function(raw_call)
        index = _object_value(raw_call, "index")
        result.append(
            {
                "index": int(index) if isinstance(index, int | float) else len(result),
                "id": _object_value(raw_call, "id"),
                "name": _object_value(function, "name"),
                "arguments": _object_value(function, "arguments"),
            }
        )
    return result


def extract_text_tool_calls(content: str) -> list[ToolCall]:
    """Parse MiMo-style text tool calls emitted as assistant content."""
    if not content or "<tool_call" not in content:
        return []
    calls: list[ToolCall] = []
    for index, block in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", content, flags=re.DOTALL)):
        name_match = re.search(r"<function=([^>]+)>", block)
        if not name_match:
            continue
        arguments: dict[str, Any] = {}
        for key, value in re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", block, flags=re.DOTALL):
            arguments[key.strip()] = _coerce_text_parameter(value.strip())
        calls.append(
            ToolCall(
                id=f"text-tool-call-{index}",
                name=name_match.group(1).strip(),
                arguments=_normalize_text_tool_arguments(arguments),
            )
        )
    return calls


def merge_tool_call_delta(buffers: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = delta["index"]
    item = buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
    if isinstance(delta.get("id"), str) and delta["id"]:
        item["id"] = delta["id"]
    if isinstance(delta.get("name"), str) and delta["name"]:
        item["name"] = delta["name"]
    if isinstance(delta.get("arguments"), str) and delta["arguments"]:
        item["arguments"] += delta["arguments"]


def buffer_to_tool_call(buffer: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=str(buffer.get("id") or "tool-call"),
        name=str(buffer.get("name") or ""),
        arguments=parse_tool_arguments(buffer.get("arguments")),
    )


def assistant_tool_call_message(
    *,
    content: str | None,
    tool_calls: list[ToolCall],
    reasoning_content: str = "",
) -> dict[str, Any]:
    payload = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in tool_calls
        ],
    }
    if reasoning_content:
        payload["reasoning_content"] = reasoning_content
    return payload


def tool_result_message(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": result.id,
        "name": result.name,
        "content": json.dumps(result.content, ensure_ascii=False),
    }


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        payload = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _tool_call_function(raw_call: Any) -> Any:
    if isinstance(raw_call, dict):
        return raw_call.get("function", {})
    return getattr(raw_call, "function", None)


def _object_value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _coerce_text_parameter(value: str) -> Any:
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


def _normalize_text_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if "limit" in normalized and "max_results" not in normalized:
        normalized["max_results"] = normalized.pop("limit")
    if "type" in normalized and "topic" not in normalized:
        normalized["topic"] = normalized.pop("type")
    return normalized
