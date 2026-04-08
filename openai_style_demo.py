import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEFAULT_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_PROMPT = "What is the capital of France?"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal OpenAI SDK style demo.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--reasoning", action="store_true", help="Enable OpenRouter reasoning")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )

    request_kwargs = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ],
    }
    if args.reasoning:
        request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    response = client.chat.completions.create(**request_kwargs)
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
