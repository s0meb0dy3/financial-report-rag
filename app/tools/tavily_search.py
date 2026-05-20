import json
import time
import urllib.error
import urllib.request
from typing import Any


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchTool:
    """Web search tool boundary used by the chat agent."""

    name = "search"
    aliases = ("tavily_search",)

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = TAVILY_SEARCH_URL,
        timeout_seconds: int = 20,
        default_max_results: int = 5,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.default_max_results = default_max_results
        self.max_retries = max(0, max_retries)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the public web for current or external information. "
                    "Use this when the answer may require up-to-date sources outside the local conversation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Maximum number of search results to return.",
                        },
                        "search_depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "Search depth. Use basic unless deeper research is needed.",
                        },
                        "topic": {
                            "type": "string",
                            "enum": ["general", "news"],
                            "description": "Search topic. Use news for current news queries.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        }

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("query must not be blank")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not set")

        max_results = _bounded_int(arguments.get("max_results"), self.default_max_results, minimum=1, maximum=10)
        search_depth = str(arguments.get("search_depth") or "basic").strip().lower()
        if search_depth not in {"basic", "advanced"}:
            search_depth = "basic"
        topic = str(arguments.get("topic") or arguments.get("type") or "general").strip().lower()
        if topic not in {"general", "news"}:
            topic = "general"

        payload = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": True,
            "include_raw_content": False,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        body = self._post_with_retries(request)

        data = json.loads(body)
        results = [_normalize_result(item) for item in data.get("results", []) if isinstance(item, dict)]
        citations = [
            {
                "doc_id": item["url"],
                "doc_name": item["title"] or item["url"],
                "page": None,
                "url": item["url"],
            }
            for item in results
            if item["url"]
        ]
        return {
            "query": query,
            "answer": data.get("answer") if isinstance(data.get("answer"), str) else "",
            "results": results,
            "citations": citations,
            "metadata": {
                "search_depth": search_depth,
                "topic": topic,
                "hit_count": len(results),
            },
        }

    def _post_with_retries(self, request: urllib.request.Request) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code < 500 or attempt >= self.max_retries:
                    raise RuntimeError(f"Tavily search failed: HTTP {exc.code} {detail[:500]}") from exc
                last_error = exc
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Tavily search failed: {exc.reason}") from exc
                last_error = exc
            time.sleep(0.35 * (attempt + 1))
        raise RuntimeError(f"Tavily search failed: {last_error}")


def _normalize_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": str(item.get("title") or ""),
        "url": str(item.get("url") or ""),
        "content": str(item.get("content") or ""),
        "score": float(item.get("score") or 0.0),
    }


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(maximum, max(minimum, parsed))
