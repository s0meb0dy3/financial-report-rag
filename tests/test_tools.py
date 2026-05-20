import unittest

from app.tools import ToolRegistry, extract_text_tool_calls


class FakeTool:
    name = "search"
    aliases = ("tavily_search",)

    def schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {"type": "object"}}}

    def run(self, arguments):
        return {
            "arguments": arguments,
            "citations": [{"doc_id": "url", "doc_name": "Title", "page": None}],
        }


class ToolRegistryTests(unittest.TestCase):
    def test_schema_exposes_each_tool_once_even_with_aliases(self) -> None:
        registry = ToolRegistry([FakeTool()])

        schemas = registry.schemas()

        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "search")

    def test_execute_supports_tool_alias(self) -> None:
        registry = ToolRegistry([FakeTool()])

        result = registry.execute(
            extract_text_tool_calls(
                "<tool_call><function=tavily_search><parameter=query>Google</parameter></function></tool_call>"
            )[0]
        )

        self.assertEqual(result.status, "done")
        self.assertEqual(result.name, "tavily_search")
        self.assertEqual(result.content["arguments"]["query"], "Google")
        self.assertEqual(result.citations[0]["doc_name"], "Title")

    def test_extract_text_tool_call_normalizes_mimo_parameters(self) -> None:
        calls = extract_text_tool_calls(
            "<tool_call> <function=search> "
            "<parameter=query>Google 最新新闻</parameter>"
            "<parameter=type>news</parameter>"
            "<parameter=limit>10</parameter>"
            "</function> </tool_call>"
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].arguments["query"], "Google 最新新闻")
        self.assertEqual(calls[0].arguments["topic"], "news")
        self.assertEqual(calls[0].arguments["max_results"], 10)


if __name__ == "__main__":
    unittest.main()
