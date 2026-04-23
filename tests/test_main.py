import unittest
from unittest.mock import patch

from main import build_arg_parser, main


class MainCliTests(unittest.TestCase):
    def test_build_arg_parser_reads_chat_subcommand_options(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["chat", "--top-k", "5", "--doc-id", "moutai", "--verbose-retrieval"])

        self.assertEqual(args.command, "chat")
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.doc_id, "moutai")
        self.assertTrue(args.verbose_retrieval)

    def test_main_dispatches_chat_command(self) -> None:
        with patch("main.run_chat_command", return_value=0) as mock_run:
            exit_code = main(["chat", "--top-k", "2"])

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0].top_k, 2)

    def test_main_dispatches_ingest_command(self) -> None:
        with patch("main.run_ingest_command", return_value=0) as mock_run:
            exit_code = main(
                [
                    "ingest",
                    "--input-dir",
                    "data/raw",
                    "--artifact-dir",
                    "data/processed/mineru",
                    "--force-parse",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0].input_dir, "data/raw")
        self.assertEqual(mock_run.call_args.args[0].artifact_dir, "data/processed/mineru")
        self.assertTrue(mock_run.call_args.args[0].force_parse)

    def test_main_dispatches_index_command(self) -> None:
        with patch("main.run_index_command", return_value=0) as mock_run:
            exit_code = main(["index", "--chunks-path", "data/processed/chunks.json"])

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0].chunks_path, "data/processed/chunks.json")

    def test_main_dispatches_eval_command(self) -> None:
        with patch("main.run_eval_command", return_value=0) as mock_run:
            exit_code = main(["eval", "--top-k", "4"])

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.args[0].top_k, 4)


if __name__ == "__main__":
    unittest.main()
