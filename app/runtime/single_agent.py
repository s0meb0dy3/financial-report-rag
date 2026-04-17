from typing import Protocol

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
                    content="不要继续调用工具。请基于已有证据直接给出最终答案；如果证据不足，就回答“我不知道”。"
                ),
            ],
            tool_definitions=self.tool_registry.get_definitions(),
        )
        updated_state = ConversationState(messages=[*messages, forced_answer])
        self.session_store.save(active_session_id, updated_state)
        return TurnResult(
            answer=forced_answer.content,
            citations=self._build_citations(tool_traces),
            tool_traces=tool_traces,
            updated_state=updated_state,
        )

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
            elif trace.tool_name == "extract_table":
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
