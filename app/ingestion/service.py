import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def build_doc_id(pdf_path: Path) -> str:
    digest = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{pdf_path.stem}-{digest}"


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


def discover_pdf_files(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.glob("*.pdf") if path.is_file())


def _write_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ingest_pdfs(pdf_paths: Iterable[Path], output_path: Path) -> list[dict[str, Any]]:
    chunks = []
    for pdf_path in pdf_paths:
        chunks.extend(extract_page_chunks(Path(pdf_path)))

    _write_chunks(chunks, output_path)
    return chunks


def ingest_pdf(pdf_path: Path, output_path: Path) -> list[dict[str, Any]]:
    return ingest_pdfs([pdf_path], output_path)


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
    return parser


def run_command(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / args.output_path
    input_dir = project_root / args.input_dir

    pdf_paths = resolve_default_pdf_files(project_root, input_dir=input_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {input_dir} or project root")

    chunks = ingest_pdfs(pdf_paths, output_path)
    print(f"Ingested {len(pdf_paths)} PDFs into {len(chunks)} chunks")
    print(f"Wrote chunks to {output_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_command(args)
