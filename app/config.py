import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MAX_CHARS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
)
from app.session import DEFAULT_SESSION_DB_PATH


DEFAULT_CHAT_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    chat_api_key: str
    chat_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_api_key: str = ""
    embedding_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_max_chars: int = DEFAULT_EMBEDDING_MAX_CHARS
    context_window_tokens: int = 128000
    chat_thinking_enabled: bool = False
    pass_reasoning_history: bool = False
    stream_include_usage: bool = True
    session_db_path: str = DEFAULT_SESSION_DB_PATH

    @classmethod
    def from_env(cls) -> "AppConfig":
        default_chat_base_url = (
            DEFAULT_MIMO_BASE_URL
            if os.environ.get("MIMO_API_KEY") and not os.environ.get("OPENROUTER_API_KEY")
            else DEFAULT_OPENROUTER_BASE_URL
        )
        chat_base_url = _first_env(
            "CHAT_BASE_URL",
            "MIMO_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENROUTER_BASE_URL",
            default=default_chat_base_url,
        )
        chat_api_key = _first_env(
            "CHAT_API_KEY",
            "MIMO_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            default="",
        )
        is_mimo = "xiaomimimo.com" in chat_base_url
        return cls(
            chat_api_key=chat_api_key,
            chat_base_url=chat_base_url,
            chat_model=os.environ.get("CHAT_MODEL", "mimo-v2.5-pro" if is_mimo else DEFAULT_CHAT_MODEL),
            embedding_api_key=_first_env(
                "EMBEDDING_API_KEY",
                "OPENROUTER_API_KEY",
                "CHAT_API_KEY",
                "MIMO_API_KEY",
                "OPENAI_API_KEY",
                default=chat_api_key,
            ),
            embedding_base_url=_first_env(
                "EMBEDDING_BASE_URL",
                "OPENROUTER_BASE_URL",
                default=DEFAULT_OPENROUTER_BASE_URL
                if os.environ.get("OPENROUTER_API_KEY")
                else chat_base_url,
            ),
            embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_batch_size=int(
                os.environ.get("EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)
            ),
            embedding_max_chars=int(
                os.environ.get("EMBEDDING_MAX_CHARS", DEFAULT_EMBEDDING_MAX_CHARS)
            ),
            context_window_tokens=int(os.environ.get("CONTEXT_WINDOW_TOKENS", "128000")),
            chat_thinking_enabled=_env_bool("CHAT_THINKING_ENABLED", default=is_mimo),
            pass_reasoning_history=_env_bool("CHAT_PASS_REASONING_HISTORY", default=is_mimo),
            stream_include_usage=_env_bool("CHAT_STREAM_INCLUDE_USAGE", default=not is_mimo),
            session_db_path=os.environ.get("SESSION_DB_PATH", DEFAULT_SESSION_DB_PATH),
        )

    def require_api_key(self) -> str:
        if not self.chat_api_key:
            raise ValueError("CHAT_API_KEY, MIMO_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY is not set")
        return self.chat_api_key

    def require_embedding_api_key(self) -> str:
        if not self.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY or OPENROUTER_API_KEY is not set")
        return self.embedding_api_key


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
