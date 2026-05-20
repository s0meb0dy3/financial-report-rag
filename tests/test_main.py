import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from main import build_arg_parser, main


class MainCliTests(unittest.TestCase):
    def test_build_arg_parser_exposes_minimal_commands(self) -> None:
        parser = build_arg_parser()

        commands = parser._subparsers._actions[1].choices.keys()

        self.assertEqual(set(commands), {"serve", "chat"})

    def test_main_dispatches_chat_command(self) -> None:
        service = MagicMock()
        service.ask.return_value = SimpleNamespace(answer="测试回答。")
        with patch("main.build_chat_service_from_env", return_value=service):
            with redirect_stdout(StringIO()) as output:
                exit_code = main(["chat", "你好", "--session-id", "session-1"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().strip(), "测试回答。")
        service.ask.assert_called_once_with("你好", session_id="session-1")
        service.close.assert_called_once()

    def test_main_dispatches_serve_command(self) -> None:
        with patch("main.uvicorn.run") as run:
            exit_code = main(["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

        self.assertEqual(exit_code, 0)
        run.assert_called_once_with("app.api:app", host="0.0.0.0", port=9000, reload=True)


if __name__ == "__main__":
    unittest.main()
