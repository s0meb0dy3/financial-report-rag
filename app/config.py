import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.session import DEFAULT_SESSION_DB_PATH


DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
DEFAULT_CHAT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MIMO_MODEL = "mimo-v2.5-pro"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    chat_api_key: str
    chat_base_url: str = DEFAULT_CHAT_BASE_URL
    chat_model: str = DEFAULT_CHAT_MODEL
    context_window_tokens: int = 128000
    chat_thinking_enabled: bool = False
    pass_reasoning_history: bool = False
    stream_include_usage: bool = True
    session_db_path: str = DEFAULT_SESSION_DB_PATH
    tavily_api_key: str = ""
    mineru_api_key: str = ""
    tracing_enabled: bool = False
    tracing_dir: str = "logs/traces"
    tracing_log_input_messages: bool = True
    tracing_max_chars: int = 2000

    @classmethod
    def from_env(cls) -> "AppConfig":
        default_chat_base_url = _default_chat_base_url()
        chat_base_url = _first_env(
            "CHAT_BASE_URL",
            "MIMO_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENROUTER_BASE_URL",
            default=default_chat_base_url,
        )
        chat_api_key = _first_env(
            "CHAT_API_KEY",
            "DEEPSEEK_API_KEY",
            "MIMO_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            default="",
        )
        is_mimo = "xiaomimimo.com" in chat_base_url
        return cls(
            chat_api_key=chat_api_key,
            chat_base_url=chat_base_url,
            chat_model=os.environ.get("CHAT_MODEL", DEFAULT_MIMO_MODEL if is_mimo else DEFAULT_CHAT_MODEL),
            context_window_tokens=int(os.environ.get("CONTEXT_WINDOW_TOKENS", "128000")),
            chat_thinking_enabled=_env_bool("CHAT_THINKING_ENABLED", default=is_mimo),
            pass_reasoning_history=_env_bool("CHAT_PASS_REASONING_HISTORY", default=is_mimo),
            stream_include_usage=_env_bool("CHAT_STREAM_INCLUDE_USAGE", default=not is_mimo),
            session_db_path=os.environ.get("SESSION_DB_PATH", DEFAULT_SESSION_DB_PATH),
            tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
            mineru_api_key=os.environ.get("MINERU_API_KEY", ""),
            tracing_enabled=_env_bool("TRACING_ENABLED", default=False),
            tracing_dir=os.environ.get("TRACING_DIR", "logs/traces"),
            tracing_log_input_messages=_env_bool("TRACING_LOG_INPUT_MESSAGES", default=True),
            tracing_max_chars=int(os.environ.get("TRACING_MAX_CHARS", "2000")),
        )

    def require_api_key(self) -> str:
        if not self.chat_api_key:
            raise ValueError(
                "CHAT_API_KEY, DEEPSEEK_API_KEY, MIMO_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY is not set"
            )
        return self.chat_api_key


def _default_chat_base_url() -> str:
    if _any_env("CHAT_BASE_URL", "MIMO_BASE_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL"):
        return DEFAULT_CHAT_BASE_URL
    if os.environ.get("CHAT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"):
        return DEFAULT_CHAT_BASE_URL
    if os.environ.get("MIMO_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
        return DEFAULT_MIMO_BASE_URL
    if os.environ.get("OPENROUTER_API_KEY"):
        return DEFAULT_OPENROUTER_BASE_URL
    return DEFAULT_CHAT_BASE_URL


def _any_env(*names: str) -> bool:
    return any(os.environ.get(name) for name in names)


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
