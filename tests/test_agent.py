import io
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent import Agent, DEFAULT_CHAT_MODEL, build_arg_parser, main


class AgentTests(unittest.TestCase):
    def test_build_arg_parser_reads_question_and_options(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["营业总收入是多少？", "--top-k", "5", "--embeddings-path", "data/test.json"])

        self.assertEqual(args.question, "营业总收入是多少？")
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.embeddings_path, "data/test.json")

    def test_from_env_reads_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            agent = Agent.from_env()

        self.assertEqual(agent.api_key, "test-key")
        self.assertEqual(agent.chat_model, DEFAULT_CHAT_MODEL)

    def test_build_user_message_includes_pages_and_text(self) -> None:
        chunks = [
            {"page": 2, "text": "营业总收入 1,741.44 亿元。"},
            {"page": 6, "text": "净利润 862.28 亿元。"},
        ]

        message = Agent.build_user_message("营业总收入是多少？", chunks)

        self.assertIn("问题：营业总收入是多少？", message)
        self.assertIn("[Page 2]", message)
        self.assertIn("[Page 6]", message)
        self.assertIn("营业总收入 1,741.44 亿元。", message)

    def test_extract_answer_reads_chat_completion_shape(self) -> None:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "贵州茅台 2024 年营业总收入为 1,741.44 亿元。[2]"

        answer = Agent.extract_answer(response)

        self.assertEqual(answer, "贵州茅台 2024 年营业总收入为 1,741.44 亿元。[2]")

    def test_ask_uses_retriever_and_returns_citations(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"page": 2, "text": "营业总收入 1,741.44 亿元。", "score": 0.9},
            {"page": 6, "text": "净利润 862.28 亿元。", "score": 0.8},
        ]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "贵州茅台 2024 年营业总收入为 1,741.44 亿元。[2]"
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(api_key="test-key", retriever=mock_retriever, client=mock_client)

        result = agent.ask("营业总收入是多少？", embeddings_path="fake.json")

        self.assertEqual(result["question"], "营业总收入是多少？")
        self.assertIn("1,741.44 亿元", result["answer"])
        self.assertEqual(result["citations"], [2, 6])
        self.assertEqual(len(result["chunks"]), 2)
        mock_retriever.search.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

    def test_main_prints_answer_and_citations(self) -> None:
        fake_result = {
            "answer": "贵州茅台 2024 年营业总收入为 1,741.44 亿元。[2]",
            "citations": [2, 6],
        }

        mock_agent = MagicMock()
        mock_agent.ask.return_value = fake_result
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with patch("agent.Agent.from_env", return_value=mock_context):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = main(
                    [
                        "营业总收入是多少？",
                        "--top-k",
                        "2",
                        "--embeddings-path",
                        "data/processed/embeddings.json",
                    ]
                )

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("1,741.44 亿元", output)
        self.assertIn("Citations: [2, 6]", output)
        mock_agent.ask.assert_called_once_with(
            "营业总收入是多少？",
            Path("/Users/peteryao/projects/CaibaoAgent/data/processed/embeddings.json"),
            top_k=2,
        )


if __name__ == "__main__":
    unittest.main()
