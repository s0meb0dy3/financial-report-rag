import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.eval import (
    build_judge_messages,
    load_questions,
    main,
    parse_judge_result,
)


class EvalTests(unittest.TestCase):
    def test_load_questions_reads_json_file(self) -> None:
        questions = [
            {
                "id": "q001",
                "question": "公司的外文名称是什么？",
                "expected_answer": "Kweichow Moutai Co.,Ltd.",
                "aliases": ["Kweichow Moutai Co., Ltd."],
                "expected_pages": [4],
                "type": "fact",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "questions.json"
            path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

            loaded = load_questions(path)

        self.assertEqual(loaded, questions)

    def test_build_judge_messages_includes_expected_and_actual_fields(self) -> None:
        messages = build_judge_messages(
            question={
                "id": "q001",
                "question": "公司的外文名称是什么？",
                "expected_answer": "Kweichow Moutai Co.,Ltd.",
                "aliases": ["Kweichow Moutai Co., Ltd."],
                "expected_pages": [4],
                "type": "fact",
            },
            answer_result={
                "answer": "公司的外文名称是 Kweichow Moutai Co.,Ltd. [4]",
                "citations": [
                    {"doc_id": "moutai", "doc_name": "茅台2024年年度报告完整版.pdf", "page": 4},
                    {"doc_id": "moutai", "doc_name": "茅台2024年年度报告完整版.pdf", "page": 74},
                ],
            },
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("JSON", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("expected_answer", messages[1]["content"])
        self.assertIn("actual_answer", messages[1]["content"])
        self.assertIn("actual_citations", messages[1]["content"])

    def test_parse_judge_result_reads_json_response(self) -> None:
        result = parse_judge_result(
            """```json
            {"pass": true, "score": 0.9, "reason": "回答正确"}
            ```"""
        )

        self.assertEqual(result["pass"], True)
        self.assertEqual(result["score"], 0.9)
        self.assertEqual(result["reason"], "回答正确")

    def test_main_runs_evaluation_and_writes_results(self) -> None:
        questions = [
            {
                "id": "q001",
                "question": "公司的外文名称是什么？",
                "expected_answer": "Kweichow Moutai Co.,Ltd.",
                "aliases": ["Kweichow Moutai Co., Ltd."],
                "expected_pages": [4],
                "type": "fact",
            }
        ]

        ask_result = {
            "question": "公司的外文名称是什么？",
            "answer": "公司的外文名称是 Kweichow Moutai Co.,Ltd. [4]",
            "citations": [
                {"doc_id": "moutai", "doc_name": "茅台2024年年度报告完整版.pdf", "page": 4},
                {"doc_id": "moutai", "doc_name": "茅台2024年年度报告完整版.pdf", "page": 74},
            ],
        }
        judge_response = MagicMock()
        judge_response.choices = [MagicMock()]
        judge_response.choices[0].message.content = (
            '{"pass": true, "score": 1.0, "reason": "回答正确，页码合理"}'
        )

        mock_agent = MagicMock()
        mock_agent.ask.return_value = ask_result
        mock_agent.chat_model = "qwen/qwen3.6-plus-preview:free"
        mock_agent.client.chat.completions.create.return_value = judge_response

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            questions_path = Path(temp_dir) / "questions.json"
            output_path = Path(temp_dir) / "results.json"
            questions_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

            with patch("app.eval.Agent.from_env", return_value=mock_context):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main(
                        [
                            "--questions-path",
                            str(questions_path),
                            "--output-path",
                            str(output_path),
                        ]
                    )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertIn("Pass rate: 1/1", stdout.getvalue())
        self.assertIn("Overall score: 1.0000", stdout.getvalue())
        self.assertEqual(saved["summary"]["total"], 1)
        self.assertEqual(saved["summary"]["passed"], 1)
        self.assertEqual(saved["results"][0]["judge"]["score"], 1.0)
        self.assertEqual(saved["summary"]["failed_questions"], [])

    def test_main_prints_failed_question_details(self) -> None:
        questions = [
            {
                "id": "q020",
                "question": "公司在年报中提到的主要风险有哪些？",
                "expected_answer": "宏观经济风险、安全风险、舆情风险、环境保护风险",
                "aliases": [],
                "expected_pages": [22],
                "type": "risk",
            }
        ]

        ask_result = {
            "question": "公司在年报中提到的主要风险有哪些？",
            "answer": "信用风险、流动风险、汇率风险和利率风险[122]",
            "citations": [
                {"doc_id": "moutai", "doc_name": "茅台2024年年度报告完整版.pdf", "page": 2},
                {"doc_id": "moutai", "doc_name": "茅台2024年年度报告完整版.pdf", "page": 122},
            ],
        }
        judge_response = MagicMock()
        judge_response.choices = [MagicMock()]
        judge_response.choices[0].message.content = (
            '{"pass": false, "score": 0.0, "reason": "回答的是财务风险，不是经营风险"}'
        )

        mock_agent = MagicMock()
        mock_agent.ask.return_value = ask_result
        mock_agent.chat_model = "qwen/qwen3.6-plus-preview:free"
        mock_agent.client.chat.completions.create.return_value = judge_response

        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_agent
        mock_context.__exit__.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            questions_path = Path(temp_dir) / "questions.json"
            output_path = Path(temp_dir) / "results.json"
            questions_path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")

            with patch("app.eval.Agent.from_env", return_value=mock_context):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    exit_code = main(
                        [
                            "--questions-path",
                            str(questions_path),
                            "--output-path",
                            str(output_path),
                        ]
                    )

            saved = json.loads(output_path.read_text(encoding="utf-8"))

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Overall score: 0.0000", output)
        self.assertIn("Failed questions:", output)
        self.assertIn("q020", output)
        self.assertIn("信用风险、流动风险、汇率风险和利率风险", output)
        self.assertIn("宏观经济风险、安全风险、舆情风险、环境保护风险", output)
        self.assertIn("回答的是财务风险，不是经营风险", output)
        self.assertEqual(len(saved["summary"]["failed_questions"]), 1)
        self.assertEqual(saved["summary"]["failed_questions"][0]["id"], "q020")


if __name__ == "__main__":
    unittest.main()
