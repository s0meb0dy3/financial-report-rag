from typing import Any, Optional

from app.domain import Citation, ToolTrace, TurnResult


ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_GRAY = "\033[90m"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def print_system(message: str) -> None:
    print(colorize(f"SYSTEM: {message}", ANSI_GRAY))


def print_assistant(message: str) -> None:
    print(colorize(f"ASSISTANT: {message}", ANSI_GREEN))


def print_error(message: str) -> None:
    print(colorize(f"ERROR: {message}", ANSI_RED))


def user_prompt_text() -> str:
    return colorize("USER > ", ANSI_CYAN)


def format_tool_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    return str(value)


def tool_result_count(tool_name: str, output: dict[str, Any]) -> Optional[int]:
    if tool_name == "search_reports":
        return len(output.get("results", []))
    if tool_name == "list_reports":
        return len(output.get("documents", []))
    return None


def print_tool_trace(trace: ToolTrace) -> None:
    argument_text = ", ".join(
        f"{key}={format_tool_value(value)}" for key, value in trace.arguments.items()
    )
    summary = f"TOOL: {trace.tool_name}({argument_text})"
    result_count = tool_result_count(trace.tool_name, trace.output)
    if result_count is not None:
        summary += f" -> {result_count} results"
    print(colorize(summary, ANSI_YELLOW))


def citations_from_turn_result(result: TurnResult) -> list[Citation]:
    return result.citations
