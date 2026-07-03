import contextvars
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool = False
    dir: str = "logs/traces"
    log_input_messages: bool = True
    max_chars: int = 2000


class Tracer:
    """Per-request trace recorder. Writes to JSONL file + stderr."""

    def __init__(
        self,
        trace_id: str,
        session_id: str,
        model: str,
        config: TracingConfig,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.model = model
        self.config = config
        self.start_time = time.monotonic()
        self.round_count = 0
        self._final_answer_len = 0

        os.makedirs(config.dir, exist_ok=True)
        self._jsonl_path = os.path.join(config.dir, f"{trace_id}.jsonl")
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

        self._logger = logging.getLogger(f"trace.{trace_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_TerminalFormatter(trace_id))
            self._logger.addHandler(handler)

    def close(self) -> None:
        self._jsonl_file.close()

    def start(self, messages: list[dict[str, Any]]) -> None:
        data: dict[str, Any] = {
            "type": "trace_start",
            "model": self.model,
            "session_id": self.session_id,
            "message_count": len(messages),
        }
        if self.config.log_input_messages:
            data["messages"] = _truncate_deep(messages, self.config.max_chars)
        self._write("TRACE_START", f"model={self.model} session={self.session_id} msgs={len(messages)}", data)

    def round_start(self, round_num: int) -> None:
        self.round_count = round_num + 1
        self._write("ROUND", f"round={round_num}", {"type": "round_start", "round": round_num})

    def model_response(
        self,
        round_num: int,
        content: str,
        reasoning: str,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, Any] | None,
        duration_ms: float,
    ) -> None:
        self._final_answer_len = len(content) if not tool_calls else self._final_answer_len
        prompt_tok = (usage or {}).get("prompt_tokens", 0)
        comp_tok = (usage or {}).get("completion_tokens", 0)
        total_tok = prompt_tok + comp_tok
        tool_count = len(tool_calls)
        msg = f"round={round_num} {prompt_tok}+{comp_tok}={total_tok}tok tools={tool_count} {duration_ms:.0f}ms"
        data: dict[str, Any] = {
            "type": "model_response",
            "round": round_num,
            "content": _truncate(content, self.config.max_chars),
            "reasoning": _truncate(reasoning, self.config.max_chars),
            "tool_calls": tool_calls,
            "usage": usage,
            "duration_ms": round(duration_ms, 1),
        }
        self._write("MODEL_RESP", msg, data)

    def tool_call_start(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        args_str = json.dumps(arguments, ensure_ascii=False)
        args_preview = _truncate(args_str, 120)
        self._write("TOOL_START", f"{name}({args_preview})", {
            "type": "tool_call_start",
            "call_id": call_id,
            "name": name,
            "arguments": _truncate_deep(arguments, self.config.max_chars),
        })

    def tool_call_end(
        self,
        call_id: str,
        name: str,
        status: str,
        duration_ms: float,
        result_summary: str,
    ) -> None:
        self._write("TOOL_END", f"{name} {status} {duration_ms:.0f}ms", {
            "type": "tool_call_end",
            "call_id": call_id,
            "name": name,
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "result_summary": _truncate(result_summary, self.config.max_chars),
        })

    def trace_end(
        self,
        status: str = "ok",
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        total_ms = (time.monotonic() - self.start_time) * 1000
        data: dict[str, Any] = {
            "type": "trace_end",
            "status": status,
            "total_duration_ms": round(total_ms, 1),
            "total_rounds": self.round_count,
            "final_answer_length": self._final_answer_len,
        }
        if error_type:
            data["error_type"] = error_type
        if error_message:
            data["error_message"] = _truncate(error_message, 500)
        suffix = f" {error_type}" if error_type else ""
        msg = f"{status}{suffix} rounds={self.round_count} {self._final_answer_len}chars {total_ms:.0f}ms"
        self._write("TRACE_END", msg, data)
        self.close()

    def _write(self, event_type: str, terminal_msg: str, data: dict[str, Any]) -> None:
        data["trace_id"] = self.trace_id
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()
        self._logger.info(terminal_msg, extra={"event_type": event_type})


# --- ContextVar ---

_active_tracer: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar(
    "active_tracer", default=None
)


def get_tracer() -> Tracer | None:
    return _active_tracer.get()


def set_tracer(tracer: Tracer | None) -> contextvars.Token:
    return _active_tracer.set(tracer)


def reset_tracer(token: contextvars.Token) -> None:
    _active_tracer.reset(token)


# --- Factory ---

def create_tracer(config: TracingConfig, session_id: str, model: str) -> Tracer | None:
    if not config.enabled:
        return None
    return Tracer(
        trace_id=make_trace_id(),
        session_id=session_id,
        model=model,
        config=config,
    )


def make_trace_id() -> str:
    now = datetime.now()
    hex4 = os.urandom(2).hex()
    return f"{now:%Y%m%d}-{now:%H%M%S}-{hex4}"


# --- Helpers ---

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[{len(text)} chars total]"


def _truncate_deep(obj: Any, max_chars: int) -> Any:
    if isinstance(obj, str):
        return _truncate(obj, max_chars)
    if isinstance(obj, list):
        return [_truncate_deep(item, max_chars) for item in obj]
    if isinstance(obj, dict):
        return {k: _truncate_deep(v, max_chars) for k, v in obj.items()}
    return obj


# --- Terminal formatter ---

class _TerminalFormatter(logging.Formatter):
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"

    _EVENT_COLORS = {
        "TRACE_START": CYAN + BOLD,
        "ROUND": DIM,
        "MODEL_RESP": GREEN,
        "TOOL_START": YELLOW,
        "TOOL_END": YELLOW,
        "TRACE_END": MAGENTA + BOLD,
    }

    def __init__(self, trace_id: str):
        super().__init__()
        self._short_id = trace_id[-12:]

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        event_type = getattr(record, "event_type", "LOG")
        color = self._EVENT_COLORS.get(event_type, "")
        return (
            f"{self.DIM}{ts}{self.RESET} "
            f"{self.CYAN}[{self._short_id}]{self.RESET} "
            f"{color}{event_type:<12}{self.RESET} "
            f"{record.getMessage()}"
        )
