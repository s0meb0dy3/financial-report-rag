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


def _normalize_preview_text(text: str, max_chars: int = 90) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def _format_page_label(item: dict[str, Any]) -> str:
    page_start = item.get("page_start")
    page_end = item.get("page_end")
    page = item.get("page")
    if isinstance(page_start, int) and isinstance(page_end, int):
        if page_start == page_end:
            return f"p.{page_start}"
        return f"p.{page_start}-{page_end}"
    if isinstance(page, int):
        return f"p.{page}"
    return "p.?"


def _print_search_report_details(output: dict[str, Any]) -> None:
    retrieval_queries = output.get("retrieval_queries", [])
    if isinstance(retrieval_queries, list) and retrieval_queries:
        print(colorize("  retrieval queries:", ANSI_GRAY))
        for query in retrieval_queries:
            print(colorize(f"    - {query}", ANSI_GRAY))

    results = output.get("results", [])
    if not isinstance(results, list) or not results:
        return

    print(colorize("  top hits:", ANSI_GRAY))
    for index, item in enumerate(results[:3], start=1):
        if not isinstance(item, dict):
            continue
        section_path = item.get("section_path", [])
        section_text = " / ".join(section_path) if isinstance(section_path, list) else ""
        chunk_type = item.get("chunk_type") or "unknown"
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "n/a"
        header = (
            f"    [{index}] {_format_page_label(item)} {chunk_type} "
            f"score={score_text}"
        )
        if section_text:
            header += f" section={section_text}"
        print(colorize(header, ANSI_GRAY))
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            print(colorize(f"        {_normalize_preview_text(text)}", ANSI_GRAY))


def print_tool_trace(trace: ToolTrace, *, verbose_retrieval: bool = False) -> None:
    argument_text = ", ".join(
        f"{key}={format_tool_value(value)}" for key, value in trace.arguments.items()
    )
    summary = f"TOOL: {trace.tool_name}({argument_text})"
    result_count = tool_result_count(trace.tool_name, trace.output)
    if result_count is not None:
        summary += f" -> {result_count} results"
    print(colorize(summary, ANSI_YELLOW))
    if verbose_retrieval and trace.tool_name == "search_reports":
        _print_search_report_details(trace.output)


def citations_from_turn_result(result: TurnResult) -> list[Citation]:
    return result.citations
