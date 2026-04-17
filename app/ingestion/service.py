import argparse
import hashlib
import json
from dataclasses import replace
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
    TableRecord,
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
        tables: list[TableRecord] = []
        pdf_list = [Path(pdf_path) for pdf_path in pdf_paths]

        artifacts.docling_json_dir.mkdir(parents=True, exist_ok=True)
        if artifacts.export_markdown:
            artifacts.markdown_dir.mkdir(parents=True, exist_ok=True)

        for pdf_path in pdf_list:
            parsed_document = self.parser.parse(pdf_path)
            self._write_docling_json(parsed_document, artifacts.docling_json_dir)
            self._write_markdown(parsed_document, artifacts)
            logical_document, logical_tables = build_logical_table_artifacts(parsed_document)
            chunks.extend(self.chunk_strategy.chunk(logical_document))
            tables.extend(logical_tables)

        self._write_chunks(chunks, artifacts.chunks_path)
        if artifacts.tables_path is not None:
            self._write_tables(tables, artifacts.tables_path)
        return chunks

    @staticmethod
    def _write_chunks(chunks: list[ChunkRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_tables(tables: list[TableRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(tables, ensure_ascii=False, indent=2),
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
    chunks_path = project_root / args.output_path
    return IngestionArtifacts(
        chunks_path=chunks_path,
        tables_path=chunks_path.with_name("tables.json"),
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
        tables_path=output_path.parent / "tables.json",
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
    if artifacts.tables_path is not None:
        print(f"Wrote table index to {artifacts.tables_path}")
    print(f"Wrote Docling JSON artifacts to {artifacts.docling_json_dir}")
    if artifacts.export_markdown:
        print(f"Wrote Markdown artifacts to {artifacts.markdown_dir}")
    return 0


PREVIEW_MAX_ROWS = 4
PREVIEW_MAX_COLS = 4
ANNOTATION_HEADERS = {"附注", "注释", "注"}
STATEMENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "income_statement": (
        "利润表",
        "营业总收入",
        "营业收入",
        "营业利润",
        "净利润",
    ),
    "balance_sheet": (
        "资产负债表",
        "资产总计",
        "负债合计",
        "所有者权益",
    ),
    "cash_flow": (
        "现金流量表",
        "经营活动产生的现金流量净额",
        "投资活动产生的现金流量净额",
        "筹资活动产生的现金流量净额",
    ),
    "key_metrics": (
        "主要会计数据",
        "主要财务指标",
        "主要财务数据",
        "关键财务指标",
    ),
}


def _preview_matrix(matrix: list[list[str]]) -> list[list[str]]:
    return [
        row[:PREVIEW_MAX_COLS]
        for row in matrix[:PREVIEW_MAX_ROWS]
    ]


def _markdown_table_from_matrix(matrix: list[list[str]]) -> str:
    if not matrix:
        return ""
    width = max(len(row) for row in matrix)
    normalized_rows = [row + [""] * (width - len(row)) for row in matrix]
    header = normalized_rows[0]
    separator = ["---"] * width
    body = normalized_rows[1:] or [[]]
    markdown_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)


def _compact_text(text: str) -> str:
    return "".join(str(text).split()).casefold()


def _normalize_title_key(title: str) -> str:
    return _compact_text(title)


def _normalize_cell(cell: Any) -> str:
    return _compact_text(str(cell))


def _row_key(row: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_cell(cell) for cell in row)


def _has_annotation_column(header: list[str]) -> bool:
    return len(header) >= 2 and _normalize_cell(header[1]) in {
        _normalize_cell(value)
        for value in ANNOTATION_HEADERS
    }


def _copy_matrix(matrix: list[list[str]]) -> list[list[str]]:
    return [list(row) for row in matrix]


def _guess_statement_type(
    title: str,
    section_path: list[str],
    text: str,
) -> str | None:
    compact_title = "".join(title.split())
    if compact_title in {"目录", "contents"}:
        return None
    title_and_section = "\n".join([title, *section_path])
    compact_title_and_section = "".join(title_and_section.split())
    compact_text = "".join(text.split())
    if not compact_title_and_section and not compact_text:
        return None

    for statement_type, keywords in STATEMENT_TYPE_KEYWORDS.items():
        if compact_title_and_section and any(keyword in compact_title_and_section for keyword in keywords):
            return statement_type
    for statement_type, keywords in STATEMENT_TYPE_KEYWORDS.items():
        if compact_text and any(keyword in compact_text for keyword in keywords):
            return statement_type
    return None


def _extract_table_fragments(document: ParsedDocument) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    section_stack: list[str] = []

    for element in document.elements:
        kind = element.get("kind")
        text = str(element.get("text", "")).strip()
        if kind == "heading":
            heading = text
            if not heading:
                continue
            level = max(1, int(element.get("level", 1)))
            section_stack[:] = section_stack[: level - 1]
            section_stack.append(heading)
            continue

        if kind != "table":
            continue

        matrix = element.get("matrix") or []
        if not isinstance(matrix, list):
            matrix = []
        title = str(element.get("title", "")).strip() or (section_stack[-1] if section_stack else "")
        source_element_id = str(element.get("element_id", "")).strip()
        fragment = {
            "doc_id": document.doc_id,
            "doc_name": document.doc_name,
            "source_element_id": source_element_id,
            "title": title,
            "statement_type_guess": _guess_statement_type(title, section_stack, text),
            "section_path": list(section_stack),
            "page_start": element.get("page_start"),
            "page_end": element.get("page_end"),
            "matrix": _copy_matrix(matrix),
            "footnotes_text": str(element.get("footnotes_text", "")).strip(),
            "text": text,
            "provenance": [
                item
                for item in element.get("provenance", [])
                if isinstance(item, dict)
            ],
        }
        fragments.append(fragment)

    return fragments


def _statement_types_compatible(current: str | None, candidate: str | None) -> bool:
    if current == candidate:
        return True
    return bool(current and not candidate)


def _titles_compatible(current_title: str, candidate_title: str) -> bool:
    current_key = _normalize_title_key(current_title)
    candidate_key = _normalize_title_key(candidate_title)
    if current_key == candidate_key:
        return True
    return bool(current_key and not candidate_key)


def _align_row_to_anchor(anchor_header: list[str] | None, row: list[str]) -> list[str] | None:
    if anchor_header is None:
        return list(row)
    if len(row) == len(anchor_header):
        return list(row)
    if len(row) == len(anchor_header) - 1 and _has_annotation_column(anchor_header):
        return [*row[:1], "", *row[1:]]
    return None


def _is_equivalent_header(anchor_header: list[str] | None, row: list[str]) -> bool:
    if anchor_header is None:
        return False
    aligned = _align_row_to_anchor(anchor_header, row)
    if aligned is None:
        return False
    if _row_key(anchor_header) == _row_key(aligned):
        return True
    if len(row) == len(anchor_header) - 1 and _has_annotation_column(anchor_header):
        anchor_without_annotation = [anchor_header[0], *anchor_header[2:]]
        return _row_key(anchor_without_annotation) == _row_key(row)
    return _row_key(anchor_header) == _row_key(aligned)


def _dedupe_non_empty_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = _compact_text(normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _start_logical_table_group(fragment: dict[str, Any]) -> dict[str, Any]:
    matrix = _copy_matrix(fragment.get("matrix", []))
    source_element_id = fragment["source_element_id"]
    return {
        "doc_id": fragment["doc_id"],
        "doc_name": fragment["doc_name"],
        "title": fragment["title"],
        "statement_type_guess": fragment.get("statement_type_guess"),
        "section_path": list(fragment.get("section_path", [])),
        "page_start": fragment.get("page_start"),
        "page_end": fragment.get("page_end"),
        "matrix": matrix,
        "anchor_header": list(matrix[0]) if matrix else None,
        "footnotes": [fragment.get("footnotes_text", "")],
        "source_element_ids": [source_element_id],
        "provenance": [
            dict(item)
            for item in fragment.get("provenance", [])
            if isinstance(item, dict)
        ],
        "fragments": [
            {
                "source_element_id": source_element_id,
                "page_start": fragment.get("page_start"),
                "page_end": fragment.get("page_end"),
                "row_count": len(matrix),
            }
        ],
    }


def _can_merge_into_group(group: dict[str, Any], fragment: dict[str, Any]) -> bool:
    current_page_end = group.get("page_end")
    next_page_start = fragment.get("page_start")
    if not isinstance(current_page_end, int) or not isinstance(next_page_start, int):
        return False
    if next_page_start > current_page_end + 1:
        return False
    if list(group.get("section_path", [])) != list(fragment.get("section_path", [])):
        return False
    if not _statement_types_compatible(
        group.get("statement_type_guess"),
        fragment.get("statement_type_guess"),
    ):
        return False
    if not _titles_compatible(str(group.get("title", "")), str(fragment.get("title", ""))):
        return False
    return True


def _extend_logical_table_group(group: dict[str, Any], fragment: dict[str, Any]) -> bool:
    if not _can_merge_into_group(group, fragment):
        return False

    anchor_header = group.get("anchor_header")
    candidate_rows = _copy_matrix(fragment.get("matrix", []))
    if anchor_header and candidate_rows and _is_equivalent_header(anchor_header, candidate_rows[0]):
        candidate_rows = candidate_rows[1:]

    aligned_rows: list[list[str]] = []
    for row in candidate_rows:
        aligned = _align_row_to_anchor(anchor_header, row)
        if aligned is None:
            return False
        aligned_rows.append(aligned)

    if anchor_header is None and aligned_rows:
        group["anchor_header"] = list(aligned_rows[0])

    group["matrix"].extend(aligned_rows)
    fragment_page_end = fragment.get("page_end")
    if isinstance(fragment_page_end, int):
        current_page_end = group.get("page_end")
        if not isinstance(current_page_end, int):
            group["page_end"] = fragment_page_end
        else:
            group["page_end"] = max(current_page_end, fragment_page_end)
    group["footnotes"].append(fragment.get("footnotes_text", ""))
    group["source_element_ids"].append(fragment["source_element_id"])
    group["provenance"].extend(
        dict(item)
        for item in fragment.get("provenance", [])
        if isinstance(item, dict)
    )
    group["fragments"].append(
        {
            "source_element_id": fragment["source_element_id"],
            "page_start": fragment.get("page_start"),
            "page_end": fragment.get("page_end"),
            "row_count": len(fragment.get("matrix", [])),
        }
    )
    return True


def _finalize_logical_table_group(group: dict[str, Any]) -> tuple[TableRecord, dict[str, Any]]:
    matrix = _copy_matrix(group.get("matrix", []))
    footnotes = _dedupe_non_empty_texts(group.get("footnotes", []))
    footnotes_text = "\n".join(footnotes)
    source_element_ids = [item for item in group.get("source_element_ids", []) if item]
    first_source_element_id = source_element_ids[0] if source_element_ids else hashlib.md5(
        json.dumps(group.get("fragments", []), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    row_count = len(matrix)
    column_count = max((len(row) for row in matrix), default=0)
    record: TableRecord = {
        "table_id": f"{group['doc_id']}-logical-{first_source_element_id}",
        "doc_id": group["doc_id"],
        "doc_name": group["doc_name"],
        "title": group["title"],
        "statement_type_guess": group.get("statement_type_guess"),
        "section_path": list(group.get("section_path", [])),
        "page_start": group.get("page_start"),
        "page_end": group.get("page_end"),
        "matrix": matrix,
        "preview_matrix": _preview_matrix(matrix),
        "footnotes_text": footnotes_text,
        "text": _markdown_table_from_matrix(matrix),
        "fragments": list(group.get("fragments", [])),
        "row_count": row_count,
        "column_count": column_count,
    }
    logical_element = {
        "element_id": record["table_id"],
        "kind": "table",
        "text": record["text"],
        "page_start": record["page_start"],
        "page_end": record["page_end"],
        "provenance": [
            dict(item)
            for item in group.get("provenance", [])
            if isinstance(item, dict)
        ],
        "matrix": _copy_matrix(matrix),
        "title": record["title"],
        "footnotes_text": footnotes_text,
        "source_element_ids": source_element_ids,
    }
    return record, logical_element


def build_logical_table_artifacts(document: ParsedDocument) -> tuple[ParsedDocument, list[TableRecord]]:
    fragments = _extract_table_fragments(document)
    if not fragments:
        return document, []

    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None
    for fragment in fragments:
        if current_group is None:
            current_group = _start_logical_table_group(fragment)
            continue
        if _extend_logical_table_group(current_group, fragment):
            continue
        groups.append(current_group)
        current_group = _start_logical_table_group(fragment)
    if current_group is not None:
        groups.append(current_group)

    logical_records: list[TableRecord] = []
    first_source_to_element: dict[str, dict[str, Any]] = {}
    skipped_source_ids: set[str] = set()
    for group in groups:
        record, logical_element = _finalize_logical_table_group(group)
        logical_records.append(record)
        fragments_meta = record.get("fragments", [])
        if not fragments_meta:
            continue
        first_source_id = fragments_meta[0]["source_element_id"]
        first_source_to_element[first_source_id] = logical_element
        skipped_source_ids.update(
            fragment["source_element_id"]
            for fragment in fragments_meta[1:]
        )

    logical_elements: list[dict[str, Any]] = []
    for element in document.elements:
        if element.get("kind") != "table":
            logical_elements.append(element)
            continue
        source_element_id = str(element.get("element_id", "")).strip()
        if source_element_id in skipped_source_ids:
            continue
        replacement = first_source_to_element.get(source_element_id)
        logical_elements.append(replacement or element)

    return replace(document, elements=logical_elements), logical_records


def extract_table_records(document: ParsedDocument) -> list[TableRecord]:
    _, records = build_logical_table_artifacts(document)
    return records


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_command(args)
