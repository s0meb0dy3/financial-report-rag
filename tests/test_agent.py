import io
import unittest
from unittest.mock import MagicMock, patch

from app.agent import (
    Agent,
    ANSI_CYAN,
    ANSI_GRAY,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    build_arg_parser,
    main,
)


class AgentCliTests(unittest.TestCase):
    def test_build_arg_parser_reads_session_options(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["--top-k", "5", "--doc-id", "moutai", "--verbose-retrieval"])

        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.doc_id, "moutai")
        self.assertTrue(args.verbose_retrieval)

    def test_main_runs_repl_and_prints_colored_user_tool_and_assistant_output(self) -> None:
        fake_result = {
            "answer": "第一轮回答",
            "tool_results": [
                {
                    "tool_name": "search_reports",
                    "arguments": {"query": "营业总收入是多少？", "top_k": 2},
                    "output": {
                        "query": "营业总收入是多少？",
                        "results": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2, "text": "收入", "score": 0.91}],
                    },
                }
            ],
        }

        mock_loop = MagicMock()
        mock_loop.run_turn.return_value = fake_result

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_loop
        mock_context.__exit__.return_value = None

        with patch("app.agent.AgentLoop.from_env", return_value=mock_context):
            with patch("builtins.input", side_effect=["营业总收入是多少？", "exit"]):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main(["--top-k", "2", "--doc-id", "doc-a"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn(f"{ANSI_GRAY}SYSTEM: 输入问题开始对话，输入 exit / quit / q 结束。{ANSI_RESET}", output)
        self.assertIn(f"{ANSI_CYAN}USER > {ANSI_RESET}", output)
        self.assertIn(f"{ANSI_YELLOW}TOOL: search_reports(query='营业总收入是多少？', top_k=2) -> 1 results{ANSI_RESET}", output)
        self.assertIn(f"{ANSI_GREEN}ASSISTANT: 第一轮回答{ANSI_RESET}", output)
        self.assertNotIn("retrieval queries:", output)
        mock_loop.run_turn.assert_called_once_with("营业总收入是多少？")

    def test_main_prints_verbose_retrieval_details_when_enabled(self) -> None:
        fake_result = {
            "answer": "第一轮回答",
            "tool_results": [
                {
                    "tool_name": "search_reports",
                    "arguments": {"query": "营业总收入是多少？", "top_k": 2, "doc_id": "doc-a"},
                    "output": {
                        "query": "营业总收入是多少？",
                        "retrieval_queries": ["营业总收入是多少？", "主要会计数据 营业总收入"],
                        "results": [
                            {
                                "doc_id": "doc-a",
                                "doc_name": "doc-a.pdf",
                                "page": 2,
                                "page_start": 2,
                                "page_end": 2,
                                "chunk_type": "table",
                                "section_path": ["第一章", "主要会计数据"],
                                "text": "营业总收入 1741.44 亿元。",
                                "score": 0.1234,
                            }
                        ],
                    },
                }
            ],
        }

        mock_loop = MagicMock()
        mock_loop.run_turn.return_value = fake_result

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_loop
        mock_context.__exit__.return_value = None

        with patch("app.agent.AgentLoop.from_env", return_value=mock_context):
            with patch("builtins.input", side_effect=["营业总收入是多少？", "exit"]):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main(["--verbose-retrieval", "--doc-id", "doc-a"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("retrieval queries:", output)
        self.assertIn("- 营业总收入是多少？", output)
        self.assertIn("- 主要会计数据 营业总收入", output)
        self.assertIn("[1] p.2 table score=0.1234 section=第一章 / 主要会计数据", output)
        self.assertIn("营业总收入 1741.44 亿元。", output)

    def test_main_skips_empty_input(self) -> None:
        mock_loop = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_loop
        mock_context.__exit__.return_value = None

        with patch("app.agent.AgentLoop.from_env", return_value=mock_context):
            with patch("builtins.input", side_effect=["", "  ", "exit"]):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn(f"{ANSI_GRAY}SYSTEM: 输入问题开始对话，输入 exit / quit / q 结束。{ANSI_RESET}", stdout.getvalue())
        mock_loop.run_turn.assert_not_called()

    def test_main_keeps_loop_alive_after_turn_failure(self) -> None:
        mock_loop = MagicMock()
        mock_loop.run_turn.side_effect = [RuntimeError("rate limited"), {"answer": "恢复成功", "tool_results": []}]

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_loop
        mock_context.__exit__.return_value = None

        with patch("app.agent.AgentLoop.from_env", return_value=mock_context):
            with patch("builtins.input", side_effect=["第一问", "第二问", "quit"]):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main([])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn(f"{ANSI_RED}ERROR: rate limited{ANSI_RESET}", output)
        self.assertIn(f"{ANSI_GREEN}ASSISTANT: 恢复成功{ANSI_RESET}", output)
        self.assertEqual(mock_loop.run_turn.call_count, 2)


class AgentServiceTests(unittest.TestCase):
    def test_ask_passes_turn_overrides_without_mutating_loop_defaults(self) -> None:
        loop = MagicMock()
        loop.chat_model = "qwen/qwen3.6-plus:free"
        loop.client = object()
        loop.top_k = 3
        loop.doc_id = "default-doc"
        loop.run_turn.return_value = {
            "answer": "回答",
            "citations": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2}],
            "tool_results": [],
        }

        agent = Agent(loop)

        result = agent.ask("营业总收入是多少？", top_k=5, filters={"doc_id": "doc-a"})

        self.assertEqual(result["answer"], "回答")
        self.assertEqual(result["citations"][0]["doc_id"], "doc-a")
        loop.run_turn.assert_called_once_with("营业总收入是多少？", top_k=5, doc_id="doc-a")
        self.assertEqual(loop.top_k, 3)
        self.assertEqual(loop.doc_id, "default-doc")


if __name__ == "__main__":
    unittest.main()
