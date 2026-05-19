from dataclasses import dataclass, field
from typing import Any, Iterable

from openai import OpenAI

from app.config import DEFAULT_CHAT_MODEL
from app.session import SQLiteSessionStore


SYSTEM_PROMPT = "你是一个有帮助的助手。回答简洁、准确。"


@dataclass(frozen=True)
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    audio_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0
    context_window_tokens: int = 128000
    context_used_tokens: int = 0
    estimated: bool = False

    @property
    def context_ratio(self) -> float:
        if self.context_window_tokens <= 0:
            return 0.0
        return min(1.0, self.context_used_tokens / self.context_window_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "audio_tokens": self.audio_tokens,
            "image_tokens": self.image_tokens,
            "video_tokens": self.video_tokens,
            "context_window_tokens": self.context_window_tokens,
            "context_used_tokens": self.context_used_tokens,
            "context_ratio": self.context_ratio,
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class ChatResult:
    session_id: str
    answer: str
    reasoning_content: str = ""
    usage: UsageInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "citations": [],
            "reasoning_content": self.reasoning_content,
            "usage": self.usage.to_dict() if self.usage else None,
        }


class ChatService:
    """Simple chat flow: send question with history, stream answer, persist the turn."""

    def __init__(
        self,
        *,
        session_store: SQLiteSessionStore,
        client: OpenAI,
        model: str = DEFAULT_CHAT_MODEL,
        max_history_turns: int = 6,
        context_window_tokens: int = 128000,
        thinking_enabled: bool = False,
        pass_reasoning_history: bool = False,
        stream_include_usage: bool = True,
    ):
        self.session_store = session_store
        self.client = client
        self.model = model
        self.max_history_turns = max_history_turns
        self.context_window_tokens = max(1, context_window_tokens)
        self.thinking_enabled = thinking_enabled
        self.pass_reasoning_history = pass_reasoning_history
        self.stream_include_usage = stream_include_usage

    def ask(
        self,
        question: str,
        *,
        session_id: str = "default",
    ) -> ChatResult:
        resolved_question = _clean_question(question)
        active_session_id = session_id or "default"
        self.session_store.ensure_session(active_session_id)
        messages = self._build_messages(resolved_question, active_session_id)
        response = self.client.chat.completions.create(**self._completion_kwargs(messages))
        message = response.choices[0].message
        answer = (getattr(message, "content", "") or "").strip() or "抱歉，我无法生成回答。"
        reasoning = _extract_reasoning(message)
        usage = self._usage_for_messages(messages, response_usage=getattr(response, "usage", None))
        return self._record(
            active_session_id,
            resolved_question,
            answer,
            reasoning_content=reasoning,
            usage=usage,
        )

    def stream(
        self,
        question: str,
        *,
        session_id: str = "default",
    ) -> Iterable[dict[str, Any]]:
        resolved_question = _clean_question(question)
        active_session_id = session_id or "default"
        self.session_store.ensure_session(active_session_id)
        yield {"event": "session", "data": {"session_id": active_session_id}}
        yield {"event": "status", "data": {"message": "生成回答"}}

        messages = self._build_messages(resolved_question, active_session_id)
        yield {"event": "usage", "data": self._usage_for_messages(messages, response_usage=None).to_dict()}

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: UsageInfo | None = None
        for item in self._stream_answer(messages):
            if item["type"] == "reasoning_delta":
                reasoning_parts.append(item["content"])
                yield {"event": "reasoning_delta", "data": {"content": item["content"]}}
                continue
            if item["type"] == "answer_delta":
                answer_parts.append(item["content"])
                yield {"event": "answer_delta", "data": {"content": item["content"]}}
                continue
            if item["type"] == "usage":
                usage = self._usage_for_messages(messages, response_usage=item.get("usage"))
                yield {"event": "usage", "data": usage.to_dict()}

        answer = "".join(answer_parts).strip() or "抱歉，我无法生成回答。"
        if usage is None:
            usage = self._usage_for_messages(messages, response_usage=None)
        result = self._record(
            active_session_id,
            resolved_question,
            answer,
            reasoning_content="".join(reasoning_parts).strip(),
            usage=usage,
        )
        yield {"event": "final", "data": result.to_dict()}

    def _stream_answer(self, messages: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        stream = self.client.chat.completions.create(**self._completion_kwargs(messages, stream=True))
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if _has_usage_values(usage):
                yield {"type": "usage", "usage": usage}
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            reasoning_content = _extract_reasoning(delta) if delta is not None else ""
            if reasoning_content:
                yield {"type": "reasoning_delta", "content": reasoning_content}
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield {"type": "answer_delta", "content": content}

    def _build_messages(
        self,
        question: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in self.session_store.list_turns(session_id)[-self.max_history_turns :]:
            messages.append({"role": "user", "content": turn.user_content})
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.assistant_content,
            }
            if self.pass_reasoning_history and turn.reasoning_content:
                assistant_message["reasoning_content"] = turn.reasoning_content
            messages.append(assistant_message)
        messages.append({"role": "user", "content": question})
        return messages

    def _completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if stream:
            kwargs["stream"] = True
            if self.stream_include_usage:
                kwargs["stream_options"] = {"include_usage": True}
        if self.thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return kwargs

    def _record(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        reasoning_content: str = "",
        usage: UsageInfo | None = None,
    ) -> ChatResult:
        self.session_store.record_turn(
            session_id,
            user_content=question,
            assistant_content=answer,
            reasoning_content=reasoning_content,
            citations=[],
            tool_results=[],
            usage=usage.to_dict() if usage is not None else None,
        )
        return ChatResult(
            session_id=session_id,
            answer=answer,
            reasoning_content=reasoning_content,
            usage=usage,
        )

    def _usage_for_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        response_usage: Any,
    ) -> UsageInfo:
        estimated_prompt_tokens = _estimate_messages_tokens(messages)
        prompt_tokens = _usage_value(response_usage, "prompt_tokens") or estimated_prompt_tokens
        completion_tokens = _usage_value(response_usage, "completion_tokens")
        total_tokens = _usage_value(response_usage, "total_tokens") or (prompt_tokens + completion_tokens)
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=_usage_nested_value(
                response_usage,
                "completion_tokens_details",
                "reasoning_tokens",
            ),
            cached_tokens=_usage_nested_value(response_usage, "prompt_tokens_details", "cached_tokens"),
            audio_tokens=_usage_nested_value(response_usage, "prompt_tokens_details", "audio_tokens"),
            image_tokens=_usage_nested_value(response_usage, "prompt_tokens_details", "image_tokens"),
            video_tokens=_usage_nested_value(response_usage, "prompt_tokens_details", "video_tokens"),
            context_window_tokens=self.context_window_tokens,
            context_used_tokens=prompt_tokens,
            estimated=response_usage is None,
        )

    def close(self) -> None:
        pass


def _clean_question(question: str) -> str:
    resolved = question.strip()
    if not resolved:
        raise ValueError("question must not be blank")
    return resolved


def _extract_reasoning(message: Any) -> str:
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(message, dict):
        for name in ("reasoning_content", "reasoning"):
            value = message.get(name)
            if isinstance(value, str) and value:
                return value
    return ""


def _usage_value(usage: Any, field: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    return int(value) if isinstance(value, int | float) and value >= 0 else 0


def _usage_nested_value(usage: Any, container: str, field: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        nested = usage.get(container)
    else:
        nested = getattr(usage, container, None)
    return _usage_value(nested, field)


def _has_usage_values(usage: Any) -> bool:
    return any(
        _usage_value(usage, field) > 0
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    )


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_estimate_text_tokens(str(item.get("content", ""))) + 4 for item in messages)


def _estimate_text_tokens(text: str) -> int:
    cjk_count = sum(1 for char in text if "一" <= char <= "鿿")
    other_count = max(0, len(text) - cjk_count)
    return max(1, cjk_count + (other_count + 3) // 4)
