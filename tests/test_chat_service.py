import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.chat_service import ChatService
from app.session import SQLiteSessionStore


def make_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def make_tool_call_response() -> SimpleNamespace:
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning_content="需要搜索。",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="tavily_search",
                                arguments='{"query": "OpenAI news", "max_results": 2}',
                            ),
                        )
                    ],
                )
            )
        ],
    )


def make_text_tool_call_response() -> SimpleNamespace:
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "<tool_call> <function=search> "
                        "<parameter=query>Google 最新新闻</parameter> "
                        "<parameter=type>news</parameter> "
                        "<parameter=limit>10</parameter> "
                        "</function> </tool_call>"
                    ),
                    reasoning_content="需要搜索新闻。",
                    tool_calls=None,
                )
            )
        ],
    )


def make_stream(*parts: str):
    for part in parts:
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content=part))])


def make_tool_call_stream():
    yield SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="call-1",
                            function=SimpleNamespace(
                                name="tavily_search",
                                arguments='{"query": "OpenAI news"}',
                            ),
                        )
                    ],
                )
            )
        ],
    )


def make_text_tool_call_stream():
    yield SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="<tool_call> <function=search> <parameter=query>Google 最新新闻</parameter> ",
                    reasoning_content=None,
                    tool_calls=None,
                )
            )
        ],
    )
    yield SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content="<parameter=type>news</parameter> <parameter=limit>10</parameter> </function> </tool_call>",
                    reasoning_content=None,
                    tool_calls=None,
                )
            )
        ],
    )


def make_reasoning_stream():
    yield SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="先思考。", content=None))],
    )
    yield SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="测试"))],
    )
    yield SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        ),
        choices=[],
    )
    yield SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="回答。"))],
    )


class FakeSearchTool:
    name = "search"
    aliases = ("tavily_search",)

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }

    def run(self, arguments):
        return {
            "query": arguments["query"],
            "max_results": arguments.get("max_results"),
            "topic": arguments.get("topic"),
            "results": [{"title": "OpenAI", "url": "https://openai.com", "content": "OpenAI news", "score": 0.9}],
            "citations": [{"doc_id": "https://openai.com", "doc_name": "OpenAI", "page": None}],
        }


class ChatServiceTests(unittest.TestCase):
    def test_ask_generates_answer_and_records_turn(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response("测试回答。")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(session_store=store, client=client, model="test-model")

            result = service.ask(" 测试问题？ ", session_id="session-1")
            turns = store.list_turns("session-1")

        self.assertEqual(result.answer, "测试回答。")
        self.assertEqual(result.session_id, "session-1")
        self.assertEqual(turns[0].user_content, "测试问题？")
        self.assertEqual(turns[0].assistant_content, "测试回答。")
        self.assertEqual(turns[0].citations, [])
        self.assertEqual(client.chat.completions.create.call_args.kwargs["messages"][-1]["content"], "测试问题？")

    def test_ask_rejects_blank_question(self) -> None:
        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(session_store=store, client=MagicMock(), model="test-model")

            with self.assertRaisesRegex(ValueError, "question must not be blank"):
                service.ask("  ", session_id="session-1")

    def test_ask_includes_all_history_by_default(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response("第二轮回答")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn(
                "session-1",
                user_content="第一问",
                assistant_content="第一轮回答",
                citations=[],
            )
            store.record_turn(
                "session-1",
                user_content="第二问",
                assistant_content="第二轮回答",
                citations=[],
            )
            service = ChatService(session_store=store, client=client, model="test-model")

            service.ask("第三问", session_id="session-1")

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "第一问")
        self.assertEqual(messages[2]["content"], "第一轮回答")
        self.assertEqual(messages[3]["content"], "第二问")
        self.assertEqual(messages[4]["content"], "第二轮回答")
        self.assertEqual(messages[5]["content"], "第三问")

    def test_ask_can_cap_history_turns(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response("第三轮回答")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn("session-1", user_content="第一问", assistant_content="第一轮回答", citations=[])
            store.record_turn("session-1", user_content="第二问", assistant_content="第二轮回答", citations=[])
            service = ChatService(session_store=store, client=client, model="test-model", max_history_turns=1)

            service.ask("第三问", session_id="session-1")

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "第二问")
        self.assertEqual(messages[2]["content"], "第二轮回答")
        self.assertEqual(messages[3]["content"], "第三问")

    def test_stream_emits_minimal_events_and_records_turn(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_stream("测试", "回答。")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(session_store=store, client=client, model="test-model")

            events = list(service.stream("测试问题？", session_id="session-1"))
            turns = store.list_turns("session-1")

        self.assertEqual(
            [event["event"] for event in events],
            ["session", "status", "usage", "answer_delta", "final"],
        )
        self.assertEqual(events[-1]["data"]["answer"], "测试回答。")
        self.assertEqual(events[-1]["data"]["citations"], [])
        self.assertEqual(turns[0].assistant_content, "测试回答。")

    def test_stream_emits_reasoning_and_usage(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_reasoning_stream()

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                session_store=store,
                client=client,
                model="test-model",
                context_window_tokens=1000,
            )

            events = list(service.stream("测试问题？", session_id="session-1"))

        reasoning = [event for event in events if event["event"] == "reasoning_delta"]
        usage = [event for event in events if event["event"] == "usage"][-1]["data"]
        self.assertEqual(reasoning[0]["data"]["content"], "先思考。")
        self.assertEqual(events[-1]["data"]["reasoning_content"], "先思考。")
        self.assertEqual(usage["prompt_tokens"], 120)
        self.assertEqual(usage["reasoning_tokens"], 12)
        self.assertEqual(usage["cached_tokens"], 40)
        self.assertEqual(usage["context_window_tokens"], 1000)
        self.assertFalse(usage["estimated"])

    def test_stream_can_omit_stream_options(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_stream("测试")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                session_store=store,
                client=client,
                model="test-model",
                stream_include_usage=False,
            )

            list(service.stream("测试问题？", session_id="session-1"))

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertNotIn("stream_options", kwargs)

    def test_mimo_reasoning_history_is_passed_back_when_enabled(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = make_response("第二轮回答")

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            store.record_turn(
                "session-1",
                user_content="第一问",
                assistant_content="第一轮回答",
                reasoning_content="第一轮思考",
                citations=[],
            )
            service = ChatService(
                session_store=store,
                client=client,
                model="mimo-v2.5-pro",
                thinking_enabled=True,
                pass_reasoning_history=True,
            )

            service.ask("第二问", session_id="session-1")

        kwargs = client.chat.completions.create.call_args.kwargs
        assistant_history = kwargs["messages"][2]
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        self.assertEqual(assistant_history["role"], "assistant")
        self.assertEqual(assistant_history["reasoning_content"], "第一轮思考")

    def test_ask_executes_model_requested_tool_call(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_tool_call_response(),
            make_response("基于搜索结果回答。"),
        ]

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                session_store=store,
                client=client,
                model="test-model",
                tools=[FakeSearchTool()],
            )

            result = service.ask("查一下 OpenAI 最新消息", session_id="session-1")
            turns = store.list_turns("session-1")

        self.assertEqual(result.answer, "基于搜索结果回答。")
        self.assertEqual(result.citations[0]["doc_name"], "OpenAI")
        self.assertEqual(result.tool_results[0]["name"], "tavily_search")
        self.assertEqual(turns[0].tool_results[0]["status"], "done")
        first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(first_kwargs["tool_choice"], "auto")
        self.assertEqual(len(first_kwargs["tools"]), 1)
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-2]["reasoning_content"], "需要搜索。")

    def test_stream_emits_tool_call_and_tool_result(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_tool_call_stream(),
            make_stream("搜索", "结果。"),
        ]

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                session_store=store,
                client=client,
                model="test-model",
                tools=[FakeSearchTool()],
            )

            events = list(service.stream("查一下 OpenAI 最新消息", session_id="session-1"))

        event_names = [event["event"] for event in events]
        self.assertIn("tool_call", event_names)
        self.assertIn("tool_result", event_names)
        self.assertEqual(events[-1]["data"]["answer"], "搜索结果。")
        self.assertEqual(events[-1]["data"]["tool_results"][0]["status"], "done")

    def test_ask_executes_mimo_text_tool_call(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_text_tool_call_response(),
            make_response("Google 最新新闻如下。"),
        ]

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                session_store=store,
                client=client,
                model="mimo-v2.5-pro",
                tools=[FakeSearchTool()],
            )

            result = service.ask("Google 最新新闻", session_id="session-1")

        tool_result = result.tool_results[0]
        self.assertEqual(result.answer, "Google 最新新闻如下。")
        self.assertEqual(tool_result["name"], "search")
        self.assertEqual(tool_result["arguments"]["query"], "Google 最新新闻")
        self.assertEqual(tool_result["arguments"]["topic"], "news")
        self.assertEqual(tool_result["arguments"]["max_results"], 10)
        second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(second_messages[-2]["reasoning_content"], "需要搜索新闻。")

    def test_stream_executes_mimo_text_tool_call_without_emitting_markup(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            make_text_tool_call_stream(),
            make_stream("Google 最新新闻如下。"),
        ]

        with TemporaryDirectory() as directory:
            store = SQLiteSessionStore(Path(directory) / "sessions.sqlite3")
            service = ChatService(
                session_store=store,
                client=client,
                model="mimo-v2.5-pro",
                tools=[FakeSearchTool()],
            )

            events = list(service.stream("Google 最新新闻", session_id="session-1"))

        answer_deltas = [event["data"]["content"] for event in events if event["event"] == "answer_delta"]
        self.assertIn("tool_call", [event["event"] for event in events])
        self.assertNotIn("<tool_call>", "".join(answer_deltas))
        self.assertEqual(events[-1]["data"]["answer"], "Google 最新新闻如下。")


if __name__ == "__main__":
    unittest.main()
