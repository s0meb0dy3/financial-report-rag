import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def chunk_page_text(
    text: str,
    page_number: int,
    max_chars: int = 800,
    overlap_chars: int = 120,
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
                    "chunk_id": f"page-{page_number}-chunk-{chunk_index}",
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

    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        chunks.extend(chunk_page_text(raw_text, page_number=index))

    return chunks


def ingest_pdf(pdf_path: Path, output_path: Path) -> list[dict[str, Any]]:
    chunks = extract_page_chunks(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    pdf_path = project_root / "茅台24年年度报告.pdf"
    output_path = project_root / "data" / "processed" / "chunks.json"
    ingest_pdf(pdf_path, output_path)
    print(f"Wrote chunks to {output_path}")
