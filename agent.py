import argparse
import os
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

from retriever import DEFAULT_OPENROUTER_BASE_URL, Retriever


load_dotenv()

DEFAULT_CHAT_MODEL = "qwen/qwen3.6-plus:free"


class Agent:
    @classmethod
    def from_env(cls) -> "Agent":
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            chat_model=os.environ.get("CHAT_MODEL", DEFAULT_CHAT_MODEL),
        )

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        retriever: Optional[Retriever] = None,
        client: Optional[OpenAI] = None,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = chat_model
        self.retriever = retriever or Retriever.from_env()
        self._owns_retriever = retriever is None
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    @staticmethod
    def build_user_message(question: str, chunks: list[dict[str, Any]]) -> str:
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"[Page {chunk['page']}]\n{chunk['text']}")

        context = "\n\n".join(context_parts)
        return (
            "请严格根据下面提供的资料回答问题。\n"
            "如果资料不足以支持答案，就明确回答“我不知道”。\n"
            "回答尽量简洁，并在关键结论后标注页码，例如 [2]。\n\n"
            f"问题：{question}\n\n"
            f"资料：\n{context}"
        )

    @staticmethod
    def extract_answer(response: Any) -> str:
        choices = getattr(response, "choices", [])
        if not choices:
            raise ValueError("No chat completion choices received")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if not content:
            raise ValueError("No chat completion content received")
        return content.strip()

    def chat(self, question: str, chunks: list[dict[str, Any]]) -> str:
        user_message = self.build_user_message(question, chunks)
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个财报问答助手，只能依据提供的资料回答。",
                },
                {"role": "user", "content": user_message},
            ],
        )
        return self.extract_answer(response)

    def ask(
        self,
        question: str,
        embeddings_path: Union[str, Path],
        top_k: int = 3,
    ) -> dict[str, Any]:
        chunks = self.retriever.search(question, Path(embeddings_path), top_k=top_k)
        answer = self.chat(question, chunks)
        citations = sorted({chunk["page"] for chunk in chunks})
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "chunks": chunks,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._owns_retriever:
            self.retriever.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about the Moutai annual report.")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings.json",
        help="Path to the embeddings JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to retrieve",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    embeddings_path = project_root / args.embeddings_path

    with Agent.from_env() as agent:
        result = agent.ask(args.question, embeddings_path, top_k=args.top_k)

    print(result["answer"])
    print(f"Citations: {result['citations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
