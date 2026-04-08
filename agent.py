import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from retriever import DEFAULT_OPENROUTER_BASE_URL, MultiQueryRetriever, QueryRewriter, Retriever


load_dotenv()

DEFAULT_CHAT_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_SYSTEM_MESSAGE = "你是一个财报问答助手，只能依据提供的资料回答。"
EXIT_COMMANDS = {"exit", "quit", "q"}


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
        multi_query_retriever: Optional[MultiQueryRetriever] = None,
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
        self.multi_query_retriever = multi_query_retriever or MultiQueryRetriever(
            base_retriever=self.retriever,
            query_rewriter=QueryRewriter(
                api_key=self.api_key,
                base_url=self.base_url,
                chat_model=self.chat_model,
                client=client,
            ),
        )
        self._owns_multi_query_retriever = multi_query_retriever is None

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
            context_parts.append(
                f"[Doc: {chunk.get('doc_name', 'unknown')} | Page: {chunk['page']}]\n{chunk['text']}"
            )

        context = "\n\n".join(context_parts)
        return (
            "请严格根据下面提供的资料回答问题。\n"
            "如果资料不足以支持答案，就明确回答“我不知道”。\n"
            "回答尽量简洁，并在关键结论后标注文档名和页码。\n\n"
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

    @staticmethod
    def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        citations = []
        for chunk in chunks:
            citation = {
                "doc_id": chunk.get("doc_id", ""),
                "doc_name": chunk.get("doc_name", ""),
                "page": chunk["page"],
            }
            key = (citation["doc_id"], citation["doc_name"], citation["page"])
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
        return citations

    def create_chat_completion(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
        )
        return self.extract_answer(response)

    def ask(
        self,
        question: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        chunks, retrieval_queries = self.multi_query_retriever.search_with_queries(
            question,
            top_k=top_k,
            filters=filters,
            history_messages=None,
        )
        user_message = self.build_user_message(question, chunks)
        answer = self.create_chat_completion(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ]
        )
        citations = self.build_citations(chunks)
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "chunks": chunks,
            "retrieval_queries": retrieval_queries,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._owns_multi_query_retriever:
            self.multi_query_retriever.close()

        if self._owns_retriever:
            self.retriever.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ChatSession:
    def __init__(
        self,
        agent: Agent,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
        history_dir: str = "logs",
    ):
        self.agent = agent
        self.top_k = top_k
        self.filters = filters
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        ]
        self.turns: list[dict[str, Any]] = []
        self.started_at = datetime.now().astimezone()
        self.history_path = self._build_history_path(history_dir)
        self.save_history()

    def _build_history_path(self, history_dir: str) -> str:
        directory = Path(history_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = self.started_at.strftime("chat-%Y-%m-%d-%H%M%S.json")
        return str(directory / filename)

    def save_history(self) -> None:
        model = getattr(self.agent, "chat_model", DEFAULT_CHAT_MODEL)
        payload = {
            "started_at": self.started_at.isoformat(),
            "model": model if isinstance(model, str) else str(model),
            "top_k": self.top_k,
            "filters": self.filters,
            "messages": self.messages,
            "turns": self.turns,
        }
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def build_rewrite_history_messages(self, limit_turns: int = 3) -> list[dict[str, str]]:
        history_messages: list[dict[str, str]] = []
        for turn in self.turns[-limit_turns:]:
            history_messages.append({"role": "user", "content": turn["question"]})
            history_messages.append({"role": "assistant", "content": turn["answer"]})
        return history_messages

    def run_turn(self, question: str) -> dict[str, Any]:
        chunks, retrieval_queries = self.agent.multi_query_retriever.search_with_queries(
            question,
            top_k=self.top_k,
            filters=self.filters,
            history_messages=self.build_rewrite_history_messages(),
        )
        user_message = Agent.build_user_message(question, chunks)

        messages = [*self.messages, {"role": "user", "content": user_message}]
        response = self.agent.client.chat.completions.create(
            model=self.agent.chat_model,
            messages=messages,
        )
        answer = Agent.extract_answer(response)

        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": answer})
        citations = Agent.build_citations(chunks)
        self.turns.append(
            {
                "question": question,
                "answer": answer,
                "citations": citations,
                "retrieval_queries": retrieval_queries,
            }
        )
        self.save_history()

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "chunks": chunks,
            "retrieval_queries": retrieval_queries,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with indexed financial reports.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many chunks to retrieve for each turn",
    )
    parser.add_argument(
        "--doc-id",
        help="Optional document filter for retrieval",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    filters = {"doc_id": args.doc_id} if args.doc_id else None

    with Agent.from_env() as agent:
        session = ChatSession(agent, top_k=args.top_k, filters=filters)
        print("输入问题开始对话，输入 exit / quit / q 结束。")

        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                print()
                break

            if not question:
                continue

            if question.lower() in EXIT_COMMANDS:
                break

            try:
                result = session.run_turn(question)
            except Exception as exc:
                print(f"Error: {exc}")
                continue

            print(result["answer"])
            print(f"Citations: {result['citations']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
