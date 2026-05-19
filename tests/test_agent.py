import io
import unittest
from unittest.mock import MagicMock, patch

from app.agent import Agent, build_arg_parser, run_chat_command


class AgentTests(unittest.TestCase):
    def test_agent_ask_delegates_to_chat_service(self) -> None:
        chat_service = MagicMock()
        chat_service.model = "test-model"
        chat_service.client = object()
        chat_service.ask.return_value.to_dict.return_value = {
            "session_id": "cli",
            "answer": "回答",
            "citations": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2}],
        }
        agent = Agent(chat_service)

        result = agent.ask("营业收入是多少？", top_k=5, doc_id="doc-a")

        self.assertEqual(result["answer"], "回答")
        self.assertEqual(result["question"], "营业收入是多少？")
        chat_service.ask.assert_called_once_with(
            "营业收入是多少？",
            session_id="cli",
            top_k=5,
            doc_id="doc-a",
            doc_ids=None,
        )

    def test_build_arg_parser_reads_single_question_options(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["营业收入是多少？", "--top-k", "5"])

        self.assertEqual(args.question, "营业收入是多少？")
        self.assertEqual(args.top_k, 5)

    def test_run_chat_command_prints_answer_and_citations(self) -> None:
        mock_agent = MagicMock()
        mock_agent.ask.return_value = {
            "answer": "营业收入为 100 亿元。",
            "citations": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 8}],
        }
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with patch("app.agent.Agent.from_env", return_value=mock_context):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = run_chat_command(build_arg_parser().parse_args(["营业收入是多少？"]))

        self.assertEqual(exit_code, 0)
        self.assertIn("营业收入为 100 亿元。", stdout.getvalue())
        self.assertIn("doc-a.pdf p.8", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
