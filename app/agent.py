import argparse
import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from app.context import ContextBuilder
from app.retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    ChromaRetriever,
    HybridRetriever,
    LLMQueryRewriter,
)
from app.runtime import OpenAIChatClient, SingleAgentRuntime
from app.session import InMemorySessionStore, SessionStore
from app.shared import (
    ANSI_CYAN,
    ANSI_GRAY,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    print_assistant,
    print_error,
    print_system,
    print_tool_trace,
    user_prompt_text,
)
from app.tools import build_default_tool_registry


load_dotenv()

DEFAULT_CHAT_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_SYSTEM_MESSAGE = (
    "你是一个财报问答助手。"
    "对于表格和指标问题，优先调用 search_tables 找候选表，再在需要时调用 extract_table 读取完整表格。"
    "如果表格工具找不到，再调用 search_reports 工具检索资料。"
    "回答时只依据检索到的证据作答；如果证据不足，就明确回答“我不知道”。"
    "给出结论时尽量带上文档名和页码。"
)
MAX_TOOL_CALLS = 5
EXIT_COMMANDS = {"exit", "quit", "q"}


class AgentLoop:
    @classmethod
    def from_env(
        cls,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        session_store: SessionStore | None = None,
    ) -> "AgentLoop":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            chat_model=os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL),
            top_k=top_k,
            doc_id=doc_id,
            session_store=session_store,
        )

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        retriever=None,
        tool_registry=None,
        client: Optional[OpenAI] = None,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        session_store: SessionStore | None = None,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = chat_model
        self.top_k = top_k
        self.doc_id = doc_id
        self.max_tool_calls = max_tool_calls
        self.session_store = session_store or InMemorySessionStore()
        self._client = client
        self.retriever = retriever or HybridRetriever(
            dense_retriever=ChromaRetriever(
                api_key=self.api_key,
                base_url=self.base_url,
                embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                batch_size=int(
                    os.environ.get("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
                ),
            ),
            query_rewriter=LLMQueryRewriter(self.client, self.chat_model),
        )
        self._owns_retriever = retriever is None
        self.tool_registry = tool_registry or build_default_tool_registry(self.retriever)
        self._runtime = SingleAgentRuntime(
            llm_client=OpenAIChatClient(self.client, self.chat_model),
            tool_registry=self.tool_registry,
            session_store=self.session_store,
            context_builder=ContextBuilder(system_prompt=DEFAULT_SYSTEM_MESSAGE),
            max_tool_calls=self.max_tool_calls,
        )

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    @staticmethod
    def _prepare_tool_arguments(
        tool_name: str,
        arguments: dict[str, Any],
        *,
        top_k: int,
        doc_id: Optional[str],
    ) -> dict[str, Any]:
        prepared = dict(arguments)
        if tool_name == "search_reports":
            prepared.setdefault("top_k", top_k)
        if tool_name in {"search_reports", "search_tables", "extract_table"} and doc_id is not None:
            prepared.setdefault("doc_id", doc_id)
        return prepared

    def run_turn(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> dict[str, Any]:
        active_session_id = session_id or "default"
        resolved_top_k = self.top_k if top_k is None else top_k
        resolved_doc_id = self.doc_id if doc_id is None else doc_id
        result = self._runtime.run_turn(
            question,
            session_id=active_session_id,
            tool_argument_preparer=lambda tool_name, arguments: self._prepare_tool_arguments(
                tool_name,
                arguments,
                top_k=resolved_top_k,
                doc_id=resolved_doc_id,
            ),
        )
        payload = {
            "answer": result.answer,
            "citations": [
                {"doc_id": item.doc_id, "doc_name": item.doc_name, "page": item.page}
                for item in result.citations
            ],
            "tool_results": [
                {
                    "tool_name": trace.tool_name,
                    "arguments": trace.arguments,
                    "output": trace.output,
                    "tool_call_id": trace.tool_call_id,
                }
                for trace in result.tool_traces
            ],
        }
        record_turn = getattr(self.session_store, "record_turn", None)
        if callable(record_turn):
            record_turn(
                active_session_id,
                user_content=question,
                assistant_content=payload["answer"],
                citations=payload["citations"],
                tool_results=payload["tool_results"],
                doc_id=resolved_doc_id,
            )
        return payload

    def run_turn_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
    ):
        active_session_id = session_id or "default"
        resolved_top_k = self.top_k if top_k is None else top_k
        resolved_doc_id = self.doc_id if doc_id is None else doc_id
        for event in self._runtime.run_turn_stream(
            question,
            session_id=active_session_id,
            tool_argument_preparer=lambda tool_name, arguments: self._prepare_tool_arguments(
                tool_name,
                arguments,
                top_k=resolved_top_k,
                doc_id=resolved_doc_id,
            ),
        ):
            if event.get("event") == "final":
                final_payload = event.get("data")
                record_turn = getattr(self.session_store, "record_turn", None)
                if callable(record_turn) and isinstance(final_payload, dict):
                    record_turn(
                        active_session_id,
                        user_content=question,
                        assistant_content=final_payload.get("answer", ""),
                        citations=final_payload.get("citations", []),
                        tool_results=final_payload.get("tool_results", []),
                        doc_id=resolved_doc_id,
                    )
            yield event

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._owns_retriever:
            self.retriever.close()

    def __enter__(self) -> "AgentLoop":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class Agent:
    @classmethod
    def from_env(cls, top_k: int = 3, doc_id: Optional[str] = None) -> "Agent":
        return cls(AgentLoop.from_env(top_k=top_k, doc_id=doc_id))

    def __init__(self, loop: AgentLoop):
        self.loop = loop
        self.chat_model = loop.chat_model
        self.client = loop.client

    def ask(self, question: str, top_k: int = 3, filters: Optional[dict] = None) -> dict:
        result = self.loop.run_turn(
            question,
            top_k=top_k,
            doc_id=filters.get("doc_id") if filters else None,
        )
        return {
            "question": question,
            "answer": result["answer"],
            "citations": result["citations"],
        }

    def close(self) -> None:
        self.loop.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with indexed financial reports.",
        add_help=add_help,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to retrieve for each tool call",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional document filter for retrieval",
    )
    parser.add_argument(
        "--verbose-retrieval",
        action="store_true",
        help="Print retrieval rewrite queries and top hit metadata for search tool calls",
    )
    return parser


def run_chat_command(args: argparse.Namespace) -> int:
    with AgentLoop.from_env(top_k=args.top_k, doc_id=args.doc_id) as loop:
        print_system("输入问题开始对话，输入 exit / quit / q 结束。")

        while True:
            try:
                print(user_prompt_text(), end="")
                question = input().strip()
            except EOFError:
                print()
                break

            if not question:
                continue

            if question.lower() in EXIT_COMMANDS:
                break

            try:
                result = loop.run_turn(question)
            except Exception as exc:
                print_error(str(exc))
                continue

            for item in result.get("tool_results", []):
                from app.domain import ToolTrace

                print_tool_trace(
                    ToolTrace(
                        tool_name=item.get("tool_name", ""),
                        arguments=item.get("arguments", {}),
                        output=item.get("output", {}),
                        tool_call_id=item.get("tool_call_id", ""),
                    ),
                    verbose_retrieval=args.verbose_retrieval,
                )
            print_assistant(result["answer"])

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_chat_command(args)


__all__ = [
    "ANSI_CYAN",
    "ANSI_GRAY",
    "ANSI_GREEN",
    "ANSI_RED",
    "ANSI_RESET",
    "ANSI_YELLOW",
    "Agent",
    "AgentLoop",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_SYSTEM_MESSAGE",
    "EXIT_COMMANDS",
    "MAX_TOOL_CALLS",
    "build_arg_parser",
    "main",
    "run_chat_command",
]
