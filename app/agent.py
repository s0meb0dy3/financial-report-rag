import argparse
from typing import Any

from openai import OpenAI

from app.chat_service import ChatService
from app.config import AppConfig
from app.session import SQLiteSessionStore


def build_chat_service_from_env(
    *,
    session_store: SQLiteSessionStore | None = None,
) -> ChatService:
    config = AppConfig.from_env()
    api_key = config.require_api_key()
    client = OpenAI(base_url=config.chat_base_url, api_key=api_key)
    return ChatService(
        session_store=session_store or SQLiteSessionStore(config.session_db_path),
        client=client,
        model=config.chat_model,
        context_window_tokens=config.context_window_tokens,
        thinking_enabled=config.chat_thinking_enabled,
        pass_reasoning_history=config.pass_reasoning_history,
        stream_include_usage=config.stream_include_usage,
    )


class Agent:
    """Simple chat agent."""

    @classmethod
    def from_env(cls) -> "Agent":
        return cls(build_chat_service_from_env())

    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service

    @property
    def client(self) -> OpenAI:
        return self.chat_service.client

    @property
    def chat_model(self) -> str:
        return self.chat_service.model

    def ask(
        self,
        question: str,
        *,
        session_id: str = "cli",
    ) -> dict[str, Any]:
        result = self.chat_service.ask(question, session_id=session_id)
        payload = result.to_dict()
        payload["question"] = question
        return payload

    def close(self) -> None:
        self.chat_service.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask one question through the chat flow.",
        add_help=add_help,
    )
    parser.add_argument("question", nargs="?", help="Question to ask")
    return parser


def run_chat_command(args: argparse.Namespace) -> int:
    question = (args.question or input("Question: ")).strip()
    if not question:
        print("Question is required")
        return 1
    with Agent.from_env() as agent:
        result = agent.ask(question)
    print(result["answer"])
    return 0


__all__ = [
    "Agent",
    "build_arg_parser",
    "build_chat_service_from_env",
    "run_chat_command",
]
