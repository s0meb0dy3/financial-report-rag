import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent import (
    Agent,
    ChatSession,
    DEFAULT_CHAT_MODEL,
    build_arg_parser,
    main,
)


class AgentTests(unittest.TestCase):
    def test_default_chat_model_uses_qwen_3_6_plus_free(self) -> None:
        self.assertEqual(DEFAULT_CHAT_MODEL, "qwen/qwen3.6-plus:free")

    def test_build_arg_parser_reads_session_options(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["--top-k", "5", "--doc-id", "moutai"])

        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.doc_id, "moutai")

    def test_from_env_reads_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            agent = Agent.from_env()

        self.assertEqual(agent.api_key, "test-key")
        self.assertEqual(agent.chat_model, DEFAULT_CHAT_MODEL)

    def test_build_user_message_includes_document_names_pages_and_text(self) -> None:
        chunks = [
            {"doc_name": "doc-a.pdf", "page": 2, "text": "营业总收入 1,741.44 亿元。"},
            {"doc_name": "doc-b.pdf", "page": 6, "text": "净利润 862.28 亿元。"},
        ]

        message = Agent.build_user_message("营业总收入是多少？", chunks)

        self.assertIn("问题：营业总收入是多少？", message)
        self.assertIn("[Doc: doc-a.pdf | Page: 2]", message)
        self.assertIn("[Doc: doc-b.pdf | Page: 6]", message)
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
            {
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "source_path": "/tmp/doc-a.pdf",
                "page": 2,
                "text": "营业总收入 1,741.44 亿元。",
                "score": 0.9,
            },
            {
                "doc_id": "doc-b",
                "doc_name": "doc-b.pdf",
                "source_path": "/tmp/doc-b.pdf",
                "page": 6,
                "text": "净利润 862.28 亿元。",
                "score": 0.8,
            },
        ]

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "贵州茅台 2024 年营业总收入为 1,741.44 亿元。[2]"
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(api_key="test-key", retriever=mock_retriever, client=mock_client)

        result = agent.ask("营业总收入是多少？", top_k=3)

        self.assertEqual(result["question"], "营业总收入是多少？")
        self.assertIn("1,741.44 亿元", result["answer"])
        self.assertEqual(
            result["citations"],
            [
                {"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2},
                {"doc_id": "doc-b", "doc_name": "doc-b.pdf", "page": 6},
            ],
        )
        self.assertEqual(len(result["chunks"]), 2)
        mock_retriever.search.assert_called_once_with("营业总收入是多少？", top_k=3, filters=None)
        mock_client.chat.completions.create.assert_called_once()


class ChatSessionTests(unittest.TestCase):
    def test_session_creates_unique_history_file_and_persists_turns(self) -> None:
        agent = MagicMock()
        agent.chat_model = "test-model"
        agent.retriever.search.return_value = [
            {"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2, "text": "收入 1", "score": 0.9}
        ]

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "回答"
        agent.client.chat.completions.create.return_value = response

        with tempfile.TemporaryDirectory() as temp_dir:
            session = ChatSession(agent, top_k=5, filters={"doc_id": "doc-a"}, history_dir=temp_dir)

            self.assertTrue(session.history_path.startswith(temp_dir))
            self.assertTrue(session.history_path.endswith(".json"))
            self.assertTrue(os.path.exists(session.history_path))

            with open(session.history_path, "r", encoding="utf-8") as handle:
                initial_payload = json.load(handle)

            self.assertEqual(initial_payload["model"], "test-model")
            self.assertEqual(initial_payload["top_k"], 5)
            self.assertEqual(initial_payload["filters"], {"doc_id": "doc-a"})
            self.assertEqual(
                initial_payload["messages"],
                [{"role": "system", "content": "你是一个财报问答助手，只能依据提供的资料回答。"}],
            )
            self.assertEqual(initial_payload["turns"], [])

            result = session.run_turn("营业收入是多少？")

            with open(session.history_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(result["answer"], "回答")
            self.assertEqual(len(payload["messages"]), 3)
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertEqual(payload["messages"][1]["role"], "user")
            self.assertEqual(payload["messages"][2], {"role": "assistant", "content": "回答"})
            self.assertEqual(payload["turns"][0]["question"], "营业收入是多少？")
            self.assertEqual(payload["turns"][0]["answer"], "回答")
            self.assertEqual(
                payload["turns"][0]["citations"],
                [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2}],
            )

    def test_run_turn_appends_full_history_to_messages(self) -> None:
        agent = MagicMock()
        agent.retriever.search.side_effect = [
            [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2, "text": "收入 1", "score": 0.9}],
            [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 6, "text": "利润 2", "score": 0.8}],
        ]

        first_response = MagicMock()
        first_response.choices = [MagicMock()]
        first_response.choices[0].message.content = "第一轮回答"

        second_response = MagicMock()
        second_response.choices = [MagicMock()]
        second_response.choices[0].message.content = "第二轮回答"

        agent.client.chat.completions.create.side_effect = [first_response, second_response]

        session = ChatSession(agent, top_k=3)
        session.run_turn("第一轮问题")
        result = session.run_turn("第二轮问题")

        first_call = agent.client.chat.completions.create.call_args_list[0]
        second_call = agent.client.chat.completions.create.call_args_list[1]

        self.assertEqual(first_call.kwargs["messages"][0]["role"], "system")
        self.assertEqual(first_call.kwargs["messages"][1]["role"], "user")
        self.assertIn("问题：第一轮问题", first_call.kwargs["messages"][1]["content"])

        messages = second_call.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("问题：第一轮问题", messages[1]["content"])
        self.assertEqual(messages[2], {"role": "assistant", "content": "第一轮回答"})
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("问题：第二轮问题", messages[3]["content"])
        self.assertEqual(result["answer"], "第二轮回答")
        self.assertEqual(
            result["citations"],
            [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 6}],
        )

    def test_run_turn_keeps_retrieval_context_per_turn(self) -> None:
        agent = MagicMock()
        agent.retriever.search.return_value = [
            {"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2, "text": "收入 1", "score": 0.9}
        ]

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "回答"
        agent.client.chat.completions.create.return_value = response

        session = ChatSession(agent, top_k=5, filters={"doc_id": "doc-a"})
        session.run_turn("营业收入是多少？")

        agent.retriever.search.assert_called_once_with(
            "营业收入是多少？",
            top_k=5,
            filters={"doc_id": "doc-a"},
        )

    def test_main_runs_repl_and_prints_answer_and_citations(self) -> None:
        fake_result = {
            "answer": "第一轮回答",
            "citations": [{"doc_id": "doc-a", "doc_name": "doc-a.pdf", "page": 2}],
        }

        mock_session = MagicMock()
        mock_session.run_turn.return_value = fake_result

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with patch("agent.Agent.from_env", return_value=mock_context):
            with patch("agent.ChatSession", return_value=mock_session):
                with patch("builtins.input", side_effect=["营业总收入是多少？", "exit"]):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        exit_code = main(["--top-k", "2", "--doc-id", "doc-a"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("输入问题开始对话", output)
        self.assertIn("第一轮回答", output)
        self.assertIn("doc-a.pdf", output)
        mock_session.run_turn.assert_called_once_with("营业总收入是多少？")
        ChatSession.assert_not_called if False else None

    def test_main_skips_empty_input(self) -> None:
        mock_session = MagicMock()
        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with patch("agent.Agent.from_env", return_value=mock_context):
            with patch("agent.ChatSession", return_value=mock_session):
                with patch("builtins.input", side_effect=["", "  ", "exit"]):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("输入问题开始对话", stdout.getvalue())
        mock_session.run_turn.assert_not_called()

    def test_main_keeps_session_alive_after_turn_failure(self) -> None:
        mock_session = MagicMock()
        mock_session.run_turn.side_effect = [RuntimeError("rate limited"), {"answer": "恢复成功", "citations": []}]

        mock_agent = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with patch("agent.Agent.from_env", return_value=mock_context):
            with patch("agent.ChatSession", return_value=mock_session):
                with patch("builtins.input", side_effect=["第一问", "第二问", "quit"]):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        exit_code = main([])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Error:", output)
        self.assertIn("恢复成功", output)
        self.assertEqual(mock_session.run_turn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
