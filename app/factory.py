from openai import OpenAI

from app.chat_service import ChatService
from app.config import AppConfig
from app.session import SQLiteSessionStore
from app.tools import TavilySearchTool


def build_chat_service_from_env(
    *,
    session_store: SQLiteSessionStore | None = None,
) -> ChatService:
    config = AppConfig.from_env()
    api_key = config.require_api_key()
    client = OpenAI(base_url=config.chat_base_url, api_key=api_key)
    tools = []
    if config.tavily_api_key:
        tools.append(TavilySearchTool(api_key=config.tavily_api_key))
    return ChatService(
        session_store=session_store or SQLiteSessionStore(config.session_db_path),
        client=client,
        model=config.chat_model,
        context_window_tokens=config.context_window_tokens,
        thinking_enabled=config.chat_thinking_enabled,
        pass_reasoning_history=config.pass_reasoning_history,
        stream_include_usage=config.stream_include_usage,
        tools=tools,
    )


__all__ = [
    "build_chat_service_from_env",
]
