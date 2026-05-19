import argparse
from typing import Callable

from app.eval import build_arg_parser as build_eval_arg_parser
from app.eval import run_eval_command
from app.ingestion import build_arg_parser as build_ingest_arg_parser
from app.ingestion import run_command as run_ingest_command
from app.retrieval import build_arg_parser as build_index_arg_parser
from app.retrieval import run_command as run_index_command


def _add_command(
    subparsers: argparse._SubParsersAction,
    *,
    name: str,
    build_parser: Callable[..., argparse.ArgumentParser],
    handler,
) -> None:
    parent = build_parser(add_help=False)
    parser = subparsers.add_parser(
        name,
        parents=[parent],
        help=parent.description,
        description=parent.description,
    )
    parser.set_defaults(handler=handler)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the financial report agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_command(
        subparsers,
        name="ingest",
        build_parser=build_ingest_arg_parser,
        handler=run_ingest_command,
    )
    _add_command(
        subparsers,
        name="index",
        build_parser=build_index_arg_parser,
        handler=run_index_command,
    )
    _add_command(
        subparsers,
        name="eval",
        build_parser=build_eval_arg_parser,
        handler=run_eval_command,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
