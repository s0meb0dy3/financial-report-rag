import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from app.ingestion.chunking import StructuredDoclingChunker
from app.ingestion.parsers import DoclingPdfParser, build_doc_id
from app.ingestion.types import (
    ChunkRecord,
    ChunkStrategy,
    DocumentParser,
    IngestionArtifacts,
    ParsedDocument,
)


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def chunk_page_text(
    text: str,
    page_number: int,
    max_chars: int = 800,
    overlap_chars: int = 120,
    *,
    doc_id: str = "default-doc",
    doc_name: str = "default.pdf",
    source_path: Path = Path("default.pdf"),
) -> list[dict[str, Any]]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []

    chunks = []
    start = 0
    chunk_index = 1

    while start < len(normalized_text):
        end = min(start + max_chars, len(normalized_text))
        chunk_text = normalized_text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-page-{page_number}-chunk-{chunk_index}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "source_path": str(source_path.resolve()),
                    "page": page_number,
                    "text": chunk_text,
                }
            )
            chunk_index += 1

        if end >= len(normalized_text):
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def extract_page_chunks(pdf_path: Path) -> list[dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    chunks = []
    doc_id = build_doc_id(pdf_path)

    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        chunks.extend(
            chunk_page_text(
                raw_text,
                page_number=index,
                doc_id=doc_id,
                doc_name=pdf_path.name,
                source_path=pdf_path,
            )
        )

    return chunks


class IngestionService:
    def __init__(self, parser: DocumentParser, chunk_strategy: ChunkStrategy):
        self.parser = parser
        self.chunk_strategy = chunk_strategy

    def ingest_pdfs(self, pdf_paths: Iterable[Path], artifacts: IngestionArtifacts) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        pdf_list = [Path(pdf_path) for pdf_path in pdf_paths]

        artifacts.docling_json_dir.mkdir(parents=True, exist_ok=True)
        if artifacts.export_markdown:
            artifacts.markdown_dir.mkdir(parents=True, exist_ok=True)

        for pdf_path in pdf_list:
            parsed_document = self.parser.parse(pdf_path)
            self._write_docling_json(parsed_document, artifacts.docling_json_dir)
            self._write_markdown(parsed_document, artifacts)
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

    @staticmethod
    def _write_docling_json(document: ParsedDocument, output_dir: Path) -> None:
        output_path = output_dir / f"{document.doc_id}.json"
        output_path.write_text(
            json.dumps(document.raw_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_markdown(document: ParsedDocument, artifacts: IngestionArtifacts) -> None:
        if not artifacts.export_markdown or not document.markdown:
            return
        output_path = artifacts.markdown_dir / f"{document.doc_id}.md"
        output_path.write_text(document.markdown, encoding="utf-8")


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
    return IngestionArtifacts(
        chunks_path=project_root / args.output_path,
        docling_json_dir=project_root / args.docling_json_dir,
        markdown_dir=project_root / args.markdown_dir,
        export_markdown=not args.disable_markdown_export,
    )


def _build_chunk_strategy(args: argparse.Namespace) -> ChunkStrategy:
    if args.chunk_strategy != "structured-docling":
        raise ValueError(f"Unsupported chunk strategy: {args.chunk_strategy}")
    return StructuredDoclingChunker(max_chars=args.max_chars)


def ingest_pdfs(pdf_paths: Iterable[Path], output_path: Path) -> list[ChunkRecord]:
    service = IngestionService(
        parser=DoclingPdfParser(),
        chunk_strategy=StructuredDoclingChunker(),
    )
    output_path = Path(output_path)
    artifacts = IngestionArtifacts(
        chunks_path=output_path,
        docling_json_dir=output_path.parent / "docling",
        markdown_dir=output_path.parent / "markdown",
    )
    return service.ingest_pdfs(pdf_paths, artifacts)


def ingest_pdf(pdf_path: Path, output_path: Path) -> list[ChunkRecord]:
    return ingest_pdfs([pdf_path], output_path)


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest one or more PDF files into chunk JSON.",
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
        "--docling-json-dir",
        default="data/processed/docling",
        help="Directory to write Docling JSON artifacts",
    )
    parser.add_argument(
        "--markdown-dir",
        default="data/processed/markdown",
        help="Directory to write Markdown debug artifacts",
    )
    parser.add_argument(
        "--chunk-strategy",
        default="structured-docling",
        help="Chunk strategy to use. Currently only structured-docling is supported.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters to keep in a single chunk before splitting",
    )
    parser.add_argument(
        "--disable-markdown-export",
        action="store_true",
        help="Skip exporting Markdown debug artifacts",
    )
    return parser


def run_command(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / args.input_dir
    pdf_paths = resolve_default_pdf_files(project_root, input_dir=input_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir} or project root")

    service = IngestionService(
        parser=DoclingPdfParser(),
        chunk_strategy=_build_chunk_strategy(args),
    )
    artifacts = _build_ingestion_artifacts(project_root, args)
    chunks = service.ingest_pdfs(pdf_paths, artifacts)

    print(f"Ingested {len(pdf_paths)} PDFs into {len(chunks)} chunks")
    print(f"Wrote chunks to {artifacts.chunks_path}")
    print(f"Wrote Docling JSON artifacts to {artifacts.docling_json_dir}")
    if artifacts.export_markdown:
        print(f"Wrote Markdown artifacts to {artifacts.markdown_dir}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_command(args)
