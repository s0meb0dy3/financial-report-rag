import argparse

import uvicorn

from app.factory import build_chat_service_from_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal chatbox backend CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the FastAPI backend.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=_serve)

    chat = subparsers.add_parser("chat", help="Send one question through the same ChatService used by the API.")
    chat.add_argument("question")
    chat.add_argument("--session-id", default="cli")
    chat.set_defaults(handler=_chat)
    return parser


def _serve(args: argparse.Namespace) -> int:
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _chat(args: argparse.Namespace) -> int:
    service = build_chat_service_from_env()
    try:
        result = service.ask(args.question, session_id=args.session_id)
    finally:
        service.close()
    print(result.answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
