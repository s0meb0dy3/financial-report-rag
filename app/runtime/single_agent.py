import json
import re
from typing import Any, Protocol

from app.context import ContextBuilder
from app.domain import Citation, ConversationState, ToolTrace, TurnResult
from app.messages import (
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from app.session import InMemorySessionStore, SessionStore


class LLMClient(Protocol):
    def generate(self, messages: list, tool_definitions=None) -> AssistantMessage:
        ...


class SingleAgentRuntime:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry,
        session_store: SessionStore | None = None,
        context_builder: ContextBuilder | None = None,
        max_tool_calls: int = 5,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.session_store = session_store or InMemorySessionStore()
        self.context_builder = context_builder or ContextBuilder()
        self.max_tool_calls = max_tool_calls

    def run_turn(
        self,
        user_text: str,
        session_id: str | None = None,
        tool_argument_preparer=None,
    ) -> TurnResult:
        active_session_id = session_id or "default"
        state = self.session_store.load(active_session_id)
        messages = self.context_builder.build(state, user_text)
        tool_traces: list[ToolTrace] = []
        tool_call_count = 0

        while tool_call_count < self.max_tool_calls:
            assistant_message = self.llm_client.generate(
                messages,
                tool_definitions=self.tool_registry.get_definitions(),
            )
            if not assistant_message.tool_calls:
                assistant_message = self._clean_final_assistant_message(
                    assistant_message,
                    tool_traces,
                )
                updated_state = ConversationState(messages=[*messages, assistant_message])
                self.session_store.save(active_session_id, updated_state)
                return TurnResult(
                    answer=assistant_message.content,
                    citations=self._build_citations(tool_traces),
                    tool_traces=tool_traces,
                    updated_state=updated_state,
                )

            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                if tool_call_count >= self.max_tool_calls:
                    break
                execution_arguments = dict(tool_call.arguments)
                if tool_argument_preparer is not None:
                    execution_arguments = tool_argument_preparer(tool_call.tool_name, execution_arguments)
                output = self.tool_registry.execute(tool_call.tool_name, **execution_arguments)
                tool_traces.append(
                    ToolTrace(
                        tool_name=tool_call.tool_name,
                        arguments=execution_arguments,
                        output=output,
                        tool_call_id=tool_call.tool_call_id,
                    )
                )
                messages.append(
                    ToolResultMessage(
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.tool_call_id,
                        output=output,
                    )
                )
                tool_call_count += 1

        forced_answer = self.llm_client.generate(
            [
                *messages,
                UserMessage(
                    content="不要继续调用工具。请基于已有证据直接给出最终答案；如果证据不足，就回答“我不知道”。如果已经生成图表，不要输出 ECharts option、JSON 配置或代码块。"
                ),
            ],
            tool_definitions=self.tool_registry.get_definitions(),
        )
        forced_answer = self._clean_final_assistant_message(forced_answer, tool_traces)
        updated_state = ConversationState(messages=[*messages, forced_answer])
        self.session_store.save(active_session_id, updated_state)
        return TurnResult(
            answer=forced_answer.content,
            citations=self._build_citations(tool_traces),
            tool_traces=tool_traces,
            updated_state=updated_state,
        )

    def run_turn_stream(
        self,
        user_text: str,
        session_id: str | None = None,
        tool_argument_preparer=None,
    ):
        active_session_id = session_id or "default"
        state = self.session_store.load(active_session_id)
        messages = self.context_builder.build(state, user_text)
        tool_traces: list[ToolTrace] = []
        tool_call_count = 0

        yield {"event": "session", "data": {"session_id": active_session_id}}

        while tool_call_count < self.max_tool_calls:
            yield {"event": "status", "data": {"message": "生成工具计划"}}
            assistant_message = yield from self._generate_assistant_stream(
                messages,
                tool_definitions=self.tool_registry.get_definitions(),
            )
            if not assistant_message.tool_calls:
                assistant_message = self._clean_final_assistant_message(
                    assistant_message,
                    tool_traces,
                )
                updated_state = ConversationState(messages=[*messages, assistant_message])
                self.session_store.save(active_session_id, updated_state)
                yield {
                    "event": "final",
                    "data": self._turn_payload(
                        assistant_message.content,
                        tool_traces,
                    ),
                }
                return

            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                if tool_call_count >= self.max_tool_calls:
                    break
                execution_arguments = dict(tool_call.arguments)
                if tool_argument_preparer is not None:
                    execution_arguments = tool_argument_preparer(
                        tool_call.tool_name,
                        execution_arguments,
                    )
                yield {
                    "event": "status",
                    "data": {"message": f"调用工具 {tool_call.tool_name}"},
                }
                output = self.tool_registry.execute(tool_call.tool_name, **execution_arguments)
                trace = ToolTrace(
                    tool_name=tool_call.tool_name,
                    arguments=execution_arguments,
                    output=output,
                    tool_call_id=tool_call.tool_call_id,
                )
                tool_traces.append(trace)
                yield {
                    "event": "tool_result",
                    "data": self._tool_trace_payload(trace),
                }
                messages.append(
                    ToolResultMessage(
                        tool_name=tool_call.tool_name,
                        tool_call_id=tool_call.tool_call_id,
                        output=output,
                    )
                )
                tool_call_count += 1

        forced_messages = [
            *messages,
            UserMessage(
                content="不要继续调用工具。请基于已有证据直接给出最终答案；如果证据不足，就回答“我不知道”。如果已经生成图表，不要输出 ECharts option、JSON 配置或代码块。"
            ),
        ]
        yield {"event": "status", "data": {"message": "生成最终答案"}}
        forced_answer = yield from self._generate_assistant_stream(
            forced_messages,
            tool_definitions=None,
        )
        forced_answer = self._clean_final_assistant_message(forced_answer, tool_traces)
        updated_state = ConversationState(messages=[*messages, forced_answer])
        self.session_store.save(active_session_id, updated_state)
        yield {
            "event": "final",
            "data": self._turn_payload(forced_answer.content, tool_traces),
        }

    def _generate_assistant_stream(self, messages: list, tool_definitions=None) -> AssistantMessage:
        generate_stream = getattr(self.llm_client, "generate_stream", None)
        if not callable(generate_stream):
            assistant_message = self.llm_client.generate(messages, tool_definitions=tool_definitions)
            if assistant_message.content:
                yield {
                    "event": "answer_delta",
                    "data": {"content": assistant_message.content},
                }
            return assistant_message

        final_message = None
        for item in generate_stream(messages, tool_definitions=tool_definitions):
            item_type = item.get("type")
            if item_type == "content_delta":
                yield {
                    "event": "answer_delta",
                    "data": {"content": item.get("content", "")},
                }
            elif item_type == "message":
                final_message = item.get("message")
        if not isinstance(final_message, AssistantMessage):
            raise RuntimeError("Streaming LLM response did not include a final assistant message")
        return final_message

    @staticmethod
    def _tool_trace_payload(trace: ToolTrace) -> dict:
        return {
            "tool_name": trace.tool_name,
            "arguments": trace.arguments,
            "output": trace.output,
            "tool_call_id": trace.tool_call_id,
        }

    @classmethod
    def _clean_final_assistant_message(
        cls,
        message: AssistantMessage,
        tool_traces: list[ToolTrace],
    ) -> AssistantMessage:
        if not any(trace.tool_name == "create_chart" for trace in tool_traces):
            return message
        message.content = cls._strip_chart_option_json(message.content)
        return message

    @classmethod
    def _strip_chart_option_json(cls, content: str) -> str:
        without_fences = re.sub(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            lambda match: "" if cls._is_chart_json_text(match.group(1)) else match.group(0),
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        decoder = json.JSONDecoder()
        output: list[str] = []
        index = 0
        while index < len(without_fences):
            if without_fences[index] != "{":
                output.append(without_fences[index])
                index += 1
                continue
            try:
                parsed, end = decoder.raw_decode(without_fences[index:])
            except json.JSONDecodeError:
                output.append(without_fences[index])
                index += 1
                continue
            if cls._looks_like_echarts_option(parsed):
                index += end
                continue
            output.append(without_fences[index])
            index += 1

        cleaned = "".join(output)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @classmethod
    def _is_chart_json_text(cls, content: str) -> bool:
        try:
            return cls._looks_like_echarts_option(json.loads(content))
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _looks_like_echarts_option(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if isinstance(value.get("echarts_option"), dict):
            return True
        keys = set(value)
        return "series" in keys and bool(keys & {"title", "tooltip", "xAxis", "yAxis", "legend", "grid"})

    def _turn_payload(self, answer: str, tool_traces: list[ToolTrace]) -> dict:
        return {
            "answer": answer,
            "citations": [
                {"doc_id": item.doc_id, "doc_name": item.doc_name, "page": item.page}
                for item in self._build_citations(tool_traces)
            ],
            "tool_results": [self._tool_trace_payload(trace) for trace in tool_traces],
        }

    @staticmethod
    def _build_citations(tool_traces: list[ToolTrace]) -> list[Citation]:
        seen = set()
        citations = []
        for trace in tool_traces:
            items = []
            if trace.tool_name == "search_reports":
                items = trace.output.get("results", [])
            elif trace.tool_name == "search_tables":
                items = trace.output.get("tables", [])
            elif trace.tool_name == "get_table":
                table = trace.output.get("table")
                items = [table] if isinstance(table, dict) else []
            for item in items:
                if not isinstance(item, dict):
                    continue
                page = item.get("page")
                if page is None:
                    page = item.get("page_start")
                citation = Citation(
                    doc_id=item.get("doc_id", trace.output.get("doc_id", "")),
                    doc_name=item.get("doc_name", ""),
                    page=page,
                )
                key = (citation.doc_id, citation.doc_name, citation.page)
                if key in seen:
                    continue
                seen.add(key)
                citations.append(citation)
        return citations
