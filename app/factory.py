from openai import OpenAI

from app.chat_service import ChatService
from app.config import AppConfig
from app.documents import DocumentService
from app.session import SQLiteSessionStore
from app.tools import (
    CreateChartTool,
    ListReportsTool,
    ReadPdfPageTool,
    ReadTableOfContentsTool,
    SearchReportTextTool,
    TavilySearchTool,
)
from app.tracing import TracingConfig


def build_chat_service_from_env(
    *,
    session_store: SQLiteSessionStore | None = None,
    document_service: DocumentService | None = None,
) -> ChatService:
    config = AppConfig.from_env()
    api_key = config.require_api_key()
    client = OpenAI(base_url=config.chat_base_url, api_key=api_key)
    resolved_document_service = document_service or DocumentService(mineru_api_key=config.mineru_api_key)
    tools = [
        ListReportsTool(resolved_document_service),
        ReadTableOfContentsTool(resolved_document_service),
        SearchReportTextTool(resolved_document_service),
        ReadPdfPageTool(resolved_document_service),
    ]
    tools.append(CreateChartTool())
    if config.tavily_api_key:
        tools.append(TavilySearchTool(api_key=config.tavily_api_key))
    tracing_config = TracingConfig(
        enabled=config.tracing_enabled,
        dir=config.tracing_dir,
        log_input_messages=config.tracing_log_input_messages,
        max_chars=config.tracing_max_chars,
    )
    return ChatService(
        session_store=session_store or SQLiteSessionStore(config.session_db_path),
        client=client,
        model=config.chat_model,
        context_window_tokens=config.context_window_tokens,
        thinking_enabled=config.chat_thinking_enabled,
        pass_reasoning_history=config.pass_reasoning_history,
        stream_include_usage=config.stream_include_usage,
        tools=tools,
        tracing_config=tracing_config,
    )


__all__ = [
    "build_chat_service_from_env",
]
