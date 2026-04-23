import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from app.ingestion.chunking import StructuredMineruChunker
from app.ingestion.parsers import MineruPdfParser, build_doc_id
from app.ingestion.types import (
    ChunkRecord,
    ChunkStrategy,
    DocumentParser,
    IngestionArtifacts,
)


class IngestionService:
    def __init__(self, parser: DocumentParser, chunk_strategy: ChunkStrategy):
        self.parser = parser
        self.chunk_strategy = chunk_strategy

    def ingest_pdfs(self, pdf_paths: Iterable[Path], artifacts: IngestionArtifacts) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        pdf_list = [Path(pdf_path) for pdf_path in pdf_paths]

        for pdf_path in pdf_list:
            parsed_document = self.parser.parse(pdf_path)
            chunks.extend(self.chunk_strategy.chunk(parsed_document))

        self._write_chunks(chunks, artifacts.chunks_path)
        return chunks

    @staticmethod
    def _write_chunks(chunks: list[ChunkRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def discover_pdf_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob("*.pdf") if path.is_file())


def resolve_default_pdf_files(project_root: Path, input_dir: Optional[Path] = None) -> list[Path]:
    if input_dir is not None:
        discovered = discover_pdf_files(input_dir)
        if discovered:
            return discovered

    raw_dir = project_root / "data" / "raw"
    discovered = discover_pdf_files(raw_dir) if raw_dir.exists() else []
    if discovered:
        return discovered

    return sorted(path for path in project_root.glob("*.pdf") if path.is_file())


def _build_ingestion_artifacts(project_root: Path, args: argparse.Namespace) -> IngestionArtifacts:
    return IngestionArtifacts(chunks_path=project_root / args.output_path)


def _build_service(project_root: Path, args: argparse.Namespace) -> IngestionService:
    parser = MineruPdfParser.from_env(
        artifact_root=project_root / args.artifact_dir,
        force_parse=args.force_parse,
    )
    return IngestionService(
        parser=parser,
        chunk_strategy=StructuredMineruChunker(max_chars=args.max_chars),
    )


def _close_parser(parser: DocumentParser) -> None:
    parser_close = getattr(parser, "close", None)
    if callable(parser_close):
        parser_close()


def ingest_pdfs(
    pdf_paths: Iterable[Path],
    output_path: Path,
    *,
    artifact_dir: Path | None = None,
    force_parse: bool = False,
    max_chars: int = 1200,
) -> list[ChunkRecord]:
    output_path = Path(output_path)
    parser = MineruPdfParser.from_env(
        artifact_root=artifact_dir or output_path.parent / "mineru",
        force_parse=force_parse,
    )
    service = IngestionService(
        parser=parser,
        chunk_strategy=StructuredMineruChunker(max_chars=max_chars),
    )
    try:
        return service.ingest_pdfs(pdf_paths, IngestionArtifacts(chunks_path=output_path))
    finally:
        _close_parser(parser)


def ingest_pdf(pdf_path: Path, output_path: Path, **kwargs) -> list[ChunkRecord]:
    return ingest_pdfs([pdf_path], output_path, **kwargs)


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest one or more PDF files into chunk JSON using MinerU precision parsing.",
        add_help=add_help,
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="Directory containing PDF files. Falls back to project root PDFs if empty.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/chunks.json",
        help="Path to write the chunk JSON file",
    )
    parser.add_argument(
        "--artifact-dir",
        default="data/processed/mineru",
        help="Directory to cache MinerU parse artifacts",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters to keep in a paragraph chunk before splitting",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Bypass cached MinerU artifacts and re-run remote parsing",
    )
    return parser


def run_command(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / args.input_dir
    pdf_paths = resolve_default_pdf_files(project_root, input_dir=input_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir} or project root")

    service = _build_service(project_root, args)
    artifacts = _build_ingestion_artifacts(project_root, args)
    try:
        chunks = service.ingest_pdfs(pdf_paths, artifacts)
    finally:
        _close_parser(service.parser)

    print(f"Ingested {len(pdf_paths)} PDFs into {len(chunks)} chunks")
    print(f"Wrote chunks to {artifacts.chunks_path}")
    print(f"Cached MinerU artifacts in {project_root / args.artifact_dir}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_command(args)
