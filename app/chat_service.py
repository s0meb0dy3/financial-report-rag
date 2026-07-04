import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from openai import OpenAI

from app.config import DEFAULT_CHAT_MODEL
from app.session import SQLiteSessionStore
from app.tracing import TracingConfig, Tracer, create_tracer, set_tracer
from app.tools import (
    ToolRegistry,
    assistant_tool_call_message,
    buffer_to_tool_call,
    extract_text_tool_calls,
    extract_tool_call_deltas,
    extract_tool_calls,
    merge_tool_call_delta,
    tool_result_message,
)
from app.tools.types import ChatTool


SYSTEM_PROMPT = (
    "你是一个有帮助的财务分析助手。回答简洁、准确。"
    "如果需要读取本地财报原文页码，先用 list_reports 确认 doc_id，再用 read_toc 查看目录了解报告结构，"
    "也可以用 search_report_text 搜索关键词定位相关页，再用 read_pdf_page 读取指定页。"
    "read_toc 和 search_report_text 返回的是物理页码，可以直接传给 read_pdf_page。"
    "如果你需要当前外部信息，可以自由调用可用工具；如果没有调用工具，不要声称已经检索过外部来源。"
    "引用本地财报内容时尽量说明报告名和页码。"
    "如果用户需要可视化数据，使用 create_chart 工具生成图表。图表会在回答中直接展示。"
)


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
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str = ""
    usage: UsageInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "answer": self.answer,
            "citations": self.citations,
            "tool_results": self.tool_results,
            "reasoning_content": self.reasoning_content,
            "usage": self.usage.to_dict() if self.usage else None,
        }


class ChatService:
    """Lightweight tool-calling chat loop; tool details stay behind ToolRegistry."""

    def __init__(
        self,
        *,
        session_store: SQLiteSessionStore,
        client: OpenAI,
        model: str = DEFAULT_CHAT_MODEL,
        max_history_turns: int | None = None,
        context_window_tokens: int = 128000,
        thinking_enabled: bool = False,
        pass_reasoning_history: bool = False,
        stream_include_usage: bool = True,
        tools: list[ChatTool] | None = None,
        max_tool_rounds: int = 15,
        tracing_config: TracingConfig | None = None,
    ):
        self.session_store = session_store
        self.client = client
        self.model = model
        self.max_history_turns = max_history_turns if max_history_turns is None else max(1, max_history_turns)
        self.context_window_tokens = max(1, context_window_tokens)
        self.thinking_enabled = thinking_enabled
        self.pass_reasoning_history = pass_reasoning_history
        self.stream_include_usage = stream_include_usage
        self.tool_registry = ToolRegistry(list(tools or []))
        self.max_tool_rounds = max(1, max_tool_rounds)
        self.tracing_config = tracing_config

    def ask(
        self,
        question: str,
        *,
        session_id: str = "default",
        doc_id: str | None = None,
        visible_page: int | None = None,
    ) -> ChatResult:
        resolved_question = _clean_question(question)
        active_session_id = session_id or "default"
        self.session_store.ensure_session(active_session_id)
        messages = self._build_messages(resolved_question, active_session_id, doc_id=doc_id, visible_page=visible_page)
        tracer = create_tracer(self.tracing_config, active_session_id, self.model) if self.tracing_config else None
        set_tracer(tracer)
        try:
            if tracer:
                tracer.start(messages)
            answer, reasoning, usage, citations, tool_results = self._answer_with_tools(messages, tracer=tracer)
            if tracer:
                tracer.trace_end("ok")
            return self._record(
                active_session_id,
                resolved_question,
                answer,
                citations=citations,
                tool_results=tool_results,
                reasoning_content=reasoning,
                usage=usage,
            )
        except Exception as exc:
            if tracer:
                tracer.trace_end("error", type(exc).__name__, str(exc))
            raise
        finally:
            set_tracer(None)

    def stream(
        self,
        question: str,
        *,
        session_id: str = "default",
        doc_id: str | None = None,
        visible_page: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        resolved_question = _clean_question(question)
        active_session_id = session_id or "default"
        self.session_store.ensure_session(active_session_id)
        tracer = create_tracer(self.tracing_config, active_session_id, self.model) if self.tracing_config else None
        set_tracer(tracer)
        try:
            yield {"event": "session", "data": {"session_id": active_session_id}}
            yield {"event": "status", "data": {"message": "生成回答"}}

            messages = self._build_messages(resolved_question, active_session_id, doc_id=doc_id, visible_page=visible_page)
            if tracer:
                tracer.start(messages)
            yield {"event": "usage", "data": self._usage_for_messages(messages, response_usage=None).to_dict()}

            answer_parts: list[str] = []
            reasoning_parts: list[str] = []
            citations: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            usage: UsageInfo | None = None
            for item in self._stream_answer_with_tools(messages, tracer=tracer):
                if item["type"] == "reasoning_delta":
                    reasoning_parts.append(item["content"])
                    yield {"event": "reasoning_delta", "data": {"content": item["content"]}}
                    continue
                if item["type"] == "answer_delta":
                    answer_parts.append(item["content"])
                    yield {"event": "answer_delta", "data": {"content": item["content"]}}
                    continue
                if item["type"] == "tool_call":
                    yield {"event": "tool_call", "data": item["data"]}
                    continue
                if item["type"] == "tool_result":
                    result = item["result"]
                    tool_results.append(result.to_dict())
                    citations.extend(result.citations)
                    yield {"event": "tool_result", "data": result.to_dict()}
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
                citations=_dedupe_citations(citations),
                tool_results=tool_results,
                reasoning_content="".join(reasoning_parts).strip(),
                usage=usage,
            )
            if tracer:
                tracer.trace_end("ok")
            yield {"event": "final", "data": result.to_dict()}
        except Exception as exc:
            if tracer:
                tracer.trace_end("error", type(exc).__name__, str(exc))
            raise
        finally:
            set_tracer(None)

    def _answer_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tracer: Tracer | None = None,
    ) -> tuple[str, str, UsageInfo, list[dict[str, Any]], list[dict[str, Any]]]:
        current_messages = list(messages)
        reasoning_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        usage: UsageInfo | None = None

        for round_num in range(self.max_tool_rounds):
            if tracer:
                tracer.round_start(round_num)
            t0 = time.monotonic()
            response = self.client.chat.completions.create(**self._completion_kwargs(current_messages))
            duration_ms = (time.monotonic() - t0) * 1000
            message = response.choices[0].message
            usage = self._usage_for_messages(current_messages, response_usage=getattr(response, "usage", None))
            reasoning = _extract_reasoning(message)
            if reasoning:
                reasoning_parts.append(reasoning)
            tool_calls = extract_tool_calls(message)
            text_tool_calls = [] if tool_calls else extract_text_tool_calls(_message_content(message))
            tool_calls = tool_calls or text_tool_calls
            if tracer:
                tracer.model_response(
                    round_num,
                    _message_content(message),
                    reasoning,
                    [{"name": c.name, "arguments": c.arguments} for c in tool_calls],
                    usage.to_dict() if usage else None,
                    duration_ms,
                )
            if not tool_calls:
                answer = _message_content(message).strip() or "抱歉，我无法生成回答。"
                return answer, "\n".join(reasoning_parts).strip(), usage, _dedupe_citations(citations), tool_results

            current_messages.append(
                assistant_tool_call_message(
                    content=None if text_tool_calls else _message_content(message),
                    tool_calls=tool_calls,
                    reasoning_content=reasoning,
                )
            )
            for call in tool_calls:
                result = self.tool_registry.execute(call)
                tool_results.append(result.to_dict())
                citations.extend(result.citations)
                current_messages.append(tool_result_message(result))

        if tracer:
            tracer.round_start(self.max_tool_rounds)
        t0 = time.monotonic()
        response = self.client.chat.completions.create(**self._completion_kwargs(current_messages, include_tools=False))
        duration_ms = (time.monotonic() - t0) * 1000
        message = response.choices[0].message
        usage = self._usage_for_messages(current_messages, response_usage=getattr(response, "usage", None))
        reasoning = _extract_reasoning(message)
        if reasoning:
            reasoning_parts.append(reasoning)
        answer = _message_content(message).strip() or "工具调用已完成，但模型没有生成最终回答。"
        if tracer:
            tracer.model_response(self.max_tool_rounds, answer, reasoning, [], usage.to_dict() if usage else None, duration_ms)
        return answer, "\n".join(reasoning_parts).strip(), usage, _dedupe_citations(citations), tool_results

    def _stream_answer_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tracer: Tracer | None = None,
    ) -> Iterable[dict[str, Any]]:
        current_messages = list(messages)
        for round_num in range(self.max_tool_rounds):
            if tracer:
                tracer.round_start(round_num)
            content_parts: list[str] = []
            reasoning_round_parts: list[str] = []
            tool_call_buffers: dict[int, dict[str, Any]] = {}
            round_usage: Any = None
            t0 = time.monotonic()
            stream = self.client.chat.completions.create(**self._completion_kwargs(current_messages, stream=True))
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if _has_usage_values(usage):
                    round_usage = usage
                    yield {"type": "usage", "usage": usage}
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                reasoning_content = _extract_reasoning(delta) if delta is not None else ""
                if reasoning_content:
                    reasoning_round_parts.append(reasoning_content)
                    yield {"type": "reasoning_delta", "content": reasoning_content}
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    content_parts.append(content)
                for call_delta in extract_tool_call_deltas(delta):
                    merge_tool_call_delta(tool_call_buffers, call_delta)

            duration_ms = (time.monotonic() - t0) * 1000
            tool_calls = [buffer_to_tool_call(item) for _, item in sorted(tool_call_buffers.items())]
            text_content = "".join(content_parts)
            text_tool_calls = [] if tool_calls else extract_text_tool_calls(text_content)
            tool_calls = tool_calls or text_tool_calls
            if tracer:
                usage_info = self._usage_for_messages(current_messages, response_usage=round_usage) if round_usage else None
                tracer.model_response(
                    round_num,
                    text_content,
                    "".join(reasoning_round_parts),
                    [{"name": c.name, "arguments": c.arguments} for c in tool_calls],
                    usage_info.to_dict() if usage_info else None,
                    duration_ms,
                )
            if not tool_calls:
                if text_content:
                    yield {"type": "answer_delta", "content": text_content}
                return
            if content_parts and not text_tool_calls:
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": "我需要调用工具补充信息后再给出最终回答。",
                    }
                )
            current_messages.append(
                assistant_tool_call_message(
                    content=None,
                    tool_calls=tool_calls,
                    reasoning_content="".join(reasoning_round_parts).strip(),
                )
            )
            for call in tool_calls:
                yield {
                    "type": "tool_call",
                    "data": {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "status": "running",
                    },
                }
                result = self.tool_registry.execute(call)
                current_messages.append(tool_result_message(result))
                yield {"type": "tool_result", "result": result}

        if tracer:
            tracer.round_start(self.max_tool_rounds)
        t0 = time.monotonic()
        stream = self.client.chat.completions.create(
            **self._completion_kwargs(current_messages, stream=True, include_tools=False)
        )
        final_content_parts: list[str] = []
        final_reasoning_parts: list[str] = []
        round_usage = None
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if _has_usage_values(usage):
                round_usage = usage
                yield {"type": "usage", "usage": usage}
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            reasoning_content = _extract_reasoning(delta) if delta is not None else ""
            if reasoning_content:
                final_reasoning_parts.append(reasoning_content)
                yield {"type": "reasoning_delta", "content": reasoning_content}
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                final_content_parts.append(content)
                yield {"type": "answer_delta", "content": content}

        # Safety net: if streaming produced no content (reasoning-only response),
        # make a non-streaming call to get the actual answer.
        if not final_content_parts:
            fallback_response = self.client.chat.completions.create(
                **self._completion_kwargs(current_messages, include_tools=False)
            )
            fallback_message = fallback_response.choices[0].message
            fallback_content = _message_content(fallback_message)
            if fallback_content:
                yield {"type": "answer_delta", "content": fallback_content}
                final_content_parts.append(fallback_content)
            if _has_usage_values(getattr(fallback_response, "usage", None)):
                round_usage = getattr(fallback_response, "usage", None)

        if tracer:
            duration_ms = (time.monotonic() - t0) * 1000
            usage_info = self._usage_for_messages(current_messages, response_usage=round_usage) if round_usage else None
            tracer.model_response(
                self.max_tool_rounds,
                "".join(final_content_parts),
                "".join(final_reasoning_parts),
                [],
                usage_info.to_dict() if usage_info else None,
                duration_ms,
            )

    def _build_messages(
        self,
        question: str,
        session_id: str,
        *,
        doc_id: str | None = None,
        visible_page: int | None = None,
    ) -> list[dict[str, Any]]:
        system_content = SYSTEM_PROMPT
        if doc_id and visible_page and visible_page > 0:
            system_content += f"\n\n用户当前正在查看文档 {doc_id} 的第 {visible_page} 页。如果用户的问题与当前页相关，可以直接读取该页内容。"
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
        turns = self.session_store.list_turns(session_id)
        if self.max_history_turns is not None:
            turns = turns[-self.max_history_turns :]
        for turn in turns:
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
        include_tools: bool = True,
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
        if include_tools and self.tool_registry.has_tools:
            kwargs["tools"] = self.tool_registry.schemas()
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _record(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        citations: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        reasoning_content: str = "",
        usage: UsageInfo | None = None,
    ) -> ChatResult:
        self.session_store.record_turn(
            session_id,
            user_content=question,
            assistant_content=answer,
            reasoning_content=reasoning_content,
            citations=citations or [],
            tool_results=tool_results or [],
            usage=usage.to_dict() if usage is not None else None,
            create_session=False,
        )
        return ChatResult(
            session_id=session_id,
            answer=answer,
            citations=citations or [],
            tool_results=tool_results or [],
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


def _message_content(message: Any) -> str:
    value = getattr(message, "content", None)
    if isinstance(value, str):
        return value
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, Any]] = set()
    result: list[dict[str, Any]] = []
    for item in citations:
        key = (str(item.get("doc_id", "")), str(item.get("doc_name", "")), item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


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
