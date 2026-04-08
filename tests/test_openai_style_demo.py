import io
import os
import unittest
from unittest.mock import MagicMock, patch

from openai_style_demo import build_arg_parser, main


class OpenAIStyleDemoTests(unittest.TestCase):
    def test_build_arg_parser_reads_prompt_and_reasoning_flag(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "你好，请介绍一下自己",
                "--model",
                "google/gemma-3-27b-it",
                "--reasoning",
            ]
        )

        self.assertEqual(args.prompt, "你好，请介绍一下自己")
        self.assertEqual(args.model, "google/gemma-3-27b-it")
        self.assertTrue(args.reasoning)

    def test_main_calls_openai_style_chat_completion(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response

            with patch("openai_style_demo.OpenAI", return_value=mock_client):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main(["How many r's are in strawberry?", "--reasoning"])

        self.assertEqual(exit_code, 0)
        self.assertIn("MagicMock", stdout.getvalue())
        mock_client.chat.completions.create.assert_called_once_with(
            model="qwen/qwen3.6-plus:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "How many r's are in strawberry?",
                },
            ],
            extra_body={"reasoning": {"enabled": True}},
        )


if __name__ == "__main__":
    unittest.main()
