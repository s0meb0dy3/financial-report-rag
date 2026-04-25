from typing import Any, Optional

from openai import OpenAI

from app.messages import AssistantMessage, OpenAIMessageAdapter


class OpenAIChatClient:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def generate(
        self,
        messages,
        tool_definitions: Optional[list[dict[str, Any]]] = None,
    ) -> AssistantMessage:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=OpenAIMessageAdapter.to_openai(messages),
            tools=tool_definitions or None,
        )
        message = response.choices[0].message
        return OpenAIMessageAdapter.from_openai_response(
            content=getattr(message, "content", "") or "",
            tool_calls=self._normalize_tool_calls(getattr(message, "tool_calls", None) or []),
        )

    def generate_stream(
        self,
        messages,
        tool_definitions: Optional[list[dict[str, Any]]] = None,
    ):
        content_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=OpenAIMessageAdapter.to_openai(messages),
            tools=tool_definitions or None,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            content_delta = getattr(delta, "content", None)
            if content_delta:
                content_parts.append(content_delta)
                yield {"type": "content_delta", "content": content_delta}

            for item in getattr(delta, "tool_calls", None) or []:
                index = getattr(item, "index", 0) or 0
                tool_call = tool_call_parts.setdefault(
                    index,
                    {
                        "id": "",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                item_id = getattr(item, "id", None)
                if item_id:
                    tool_call["id"] = item_id
                function = getattr(item, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    if name:
                        tool_call["function"]["name"] += name
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        tool_call["function"]["arguments"] += arguments

        tool_calls = [
            value
            for _, value in sorted(tool_call_parts.items())
            if value["function"]["name"]
        ]
        yield {
            "type": "message",
            "message": OpenAIMessageAdapter.from_openai_response(
                content="".join(content_parts),
                tool_calls=tool_calls,
            ),
        }

    @staticmethod
    def _normalize_tool_calls(tool_calls) -> list[dict[str, Any]]:
        normalized = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                normalized.append(tool_call)
                continue
            normalized.append(
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            )
        return normalized
