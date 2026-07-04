import os
import unittest
from unittest.mock import patch

from app.config import AppConfig


class AppConfigTests(unittest.TestCase):
    def test_defaults_to_deepseek_chat_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.chat_base_url, "https://api.deepseek.com")
        self.assertEqual(config.chat_model, "deepseek-v4-flash")
        self.assertFalse(config.chat_thinking_enabled)
        self.assertFalse(config.pass_reasoning_history)
        self.assertTrue(config.stream_include_usage)
        self.assertEqual(config.mineru_api_key, "")

    def test_chat_env_takes_precedence(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CHAT_API_KEY": "chat-key",
                "CHAT_BASE_URL": "https://example.test/v1",
                "CHAT_MODEL": "custom-model",
                "MINERU_API_KEY": "mineru-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "MIMO_API_KEY": "mimo-key",
                "OPENROUTER_API_KEY": "openrouter-key",
            },
            clear=True,
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.chat_api_key, "chat-key")
        self.assertEqual(config.chat_base_url, "https://example.test/v1")
        self.assertEqual(config.chat_model, "custom-model")
        self.assertEqual(config.mineru_api_key, "mineru-key")

    def test_deepseek_key_alias_uses_default_deepseek_base_url(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-key"}, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.chat_api_key, "deepseek-key")
        self.assertEqual(config.chat_base_url, "https://api.deepseek.com")
        self.assertEqual(config.chat_model, "deepseek-v4-flash")

    def test_mimo_base_url_keeps_thinking_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MIMO_API_KEY": "mimo-key",
                "CHAT_BASE_URL": "https://api.xiaomimimo.com/v1",
            },
            clear=True,
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.chat_api_key, "mimo-key")
        self.assertEqual(config.chat_model, "mimo-v2.5-pro")
        self.assertTrue(config.chat_thinking_enabled)
        self.assertTrue(config.pass_reasoning_history)
        self.assertFalse(config.stream_include_usage)


if __name__ == "__main__":
    unittest.main()
