import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass
class BaseMessage:
    role: str
    content: str = ""
    created_at: str = field(default_factory=_now)


@dataclass
class SystemMessage(BaseMessage):
    role: str = "system"


@dataclass
class UserMessage(BaseMessage):
    role: str = "user"


@dataclass
class AssistantMessage(BaseMessage):
    role: str = "assistant"
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolCallMessage(BaseMessage):
    tool_name: str = ""
    tool_call_id: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    role: str = "assistant"


@dataclass
class ToolResultMessage(BaseMessage):
    tool_name: str = ""
    tool_call_id: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    role: str = "tool"


class OpenAIMessageAdapter:
    @staticmethod
    def to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, (SystemMessage, UserMessage)):
                payload.append({"role": message.role, "content": message.content})
                continue

            if isinstance(message, AssistantMessage):
                item: dict[str, Any] = {"role": "assistant", "content": message.content}
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tool_call.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
                payload.append(item)
                continue

            if isinstance(message, ToolCallMessage):
                payload.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": message.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": message.tool_name,
                                    "arguments": json.dumps(message.arguments, ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )
                continue

            if isinstance(message, ToolResultMessage):
                payload.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": json.dumps(message.output, ensure_ascii=False),
                    }
                )
                continue

            raise TypeError(f"Unsupported message type: {type(message)!r}")
        return payload

    @staticmethod
    def from_openai_response(
        *,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> AssistantMessage:
        parsed_tool_calls = []
        for tool_call in tool_calls or []:
            parsed_tool_calls.append(
                ToolCall(
                    tool_name=tool_call["function"]["name"],
                    arguments=json.loads(tool_call["function"]["arguments"]) if tool_call["function"].get("arguments") else {},
                    tool_call_id=tool_call["id"],
                )
            )
        return AssistantMessage(content=content or "", tool_calls=parsed_tool_calls)
