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
