import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.ingestion.types import ChunkRecord, ParsedDocument, ParsedElement


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def _coerce_provenance(element: ParsedElement) -> list[dict[str, Any]]:
    provenance = element.get("provenance") or []
    return [item for item in provenance if isinstance(item, dict)]


def _page_bounds_from_elements(elements: list[ParsedElement]) -> tuple[int | None, int | None]:
    page_numbers: list[int] = []
    for element in elements:
        page_start = element.get("page_start")
        page_end = element.get("page_end")
        if isinstance(page_start, int):
            page_numbers.append(page_start)
        if isinstance(page_end, int):
            page_numbers.append(page_end)
        for provenance in _coerce_provenance(element):
            page_no = provenance.get("page") or provenance.get("page_no")
            if isinstance(page_no, int):
                page_numbers.append(page_no)
    if not page_numbers:
        return None, None
    return min(page_numbers), max(page_numbers)


def _merge_provenance(elements: list[ParsedElement]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    provenance: list[dict[str, Any]] = []
    bbox_refs: list[dict[str, Any]] = []
    for element in elements:
        for item in _coerce_provenance(element):
            provenance.append(item)
            bbox = item.get("bbox")
            if bbox is not None:
                bbox_refs.append({"page": item.get("page") or item.get("page_no"), "bbox": bbox})
    return provenance, bbox_refs


def _split_text_window(text: str, max_chars: int) -> list[str]:
    normalized = _normalize_text(text)
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    split_patterns = ["\n", "。", "！", "？", "；", ".", ";", ","]
    pieces: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            for pattern in split_patterns:
                split_at = normalized.rfind(pattern, start, end)
                if split_at > start:
                    end = split_at + len(pattern)
                    break
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = max(end, start + 1)
    return pieces


def _normalize_cell_text(text: str) -> str:
    lowered = re.sub(r"\s+", "", text).strip().lower()
    return re.sub(r"[|:：()（）]", "", lowered)


def _first_non_empty_row(rows: list[list[str]]) -> list[str]:
    for row in rows:
        normalized = [cell for cell in row if _normalize_cell_text(cell)]
        if normalized:
            return row
    return []


def _rows_match(left: list[str], right: list[str]) -> bool:
    left_signature = [_normalize_cell_text(item) for item in left if _normalize_cell_text(item)]
    right_signature = [_normalize_cell_text(item) for item in right if _normalize_cell_text(item)]
    if not left_signature or not right_signature:
        return False
    if left_signature == right_signature:
        return True
    left_joined = "|".join(left_signature)
    right_joined = "|".join(right_signature)
    return left_joined in right_joined or right_joined in left_joined


def _build_section_prefix(section_path: list[str]) -> str:
    cleaned = [item.strip() for item in section_path if item and item.strip()]
    return " > ".join(cleaned)


def _build_embedding_text(section_path: list[str], text: str) -> str:
    normalized = _normalize_text(text)
    prefix = _build_section_prefix(section_path)
    if not prefix:
        return normalized
    if not normalized:
        return prefix
    return f"{prefix}\n\n{normalized}"


def _collect_table_text(
    rows: Iterable[Iterable[str]],
    *,
    captions: list[str] | None = None,
    footnotes: list[str] | None = None,
) -> str:
    sections: list[str] = []
    if captions:
        sections.extend(item for item in captions if item)
    body_rows = []
    for row in rows:
        cells = [_normalize_text(cell) for cell in row if _normalize_text(cell)]
        if cells:
            body_rows.append(" | ".join(cells))
    if body_rows:
        sections.append("\n".join(body_rows))
    if footnotes:
        sections.extend(item for item in footnotes if item)
    return _normalize_text("\n".join(sections))


def _build_table_html(row_html: list[str], fallback_html: list[str]) -> str:
    if row_html:
        return "<table>" + "".join(row_html) + "</table>"
    return "\n".join(item for item in fallback_html if item)


def _can_merge_tables(current: ParsedElement, next_element: ParsedElement) -> bool:
    current_page_end = current.get("page_end")
    next_page_start = next_element.get("page_start")
    if not isinstance(current_page_end, int) or not isinstance(next_page_start, int):
        return False
    if next_page_start != current_page_end + 1:
        return False

    current_column_count = int(current.get("table_column_count", 0) or 0)
    next_column_count = int(next_element.get("table_column_count", 0) or 0)
    if current_column_count <= 0 or current_column_count != next_column_count:
        return False

    if current.get("table_continuation_hint") or next_element.get("table_continuation_hint"):
        return True

    current_header = _first_non_empty_row(current.get("table_rows") or [])
    next_header = _first_non_empty_row(next_element.get("table_rows") or [])
    return _rows_match(current_header, next_header)


def _merge_table_elements(elements: list[ParsedElement]) -> dict[str, Any]:
    if not elements:
        return {"text": "", "embedding_text": "", "table_html": ""}

    merged_rows: list[list[str]] = []
    merged_row_html: list[str] = []
    fallback_html: list[str] = []
    captions: list[str] = []
    footnotes: list[str] = []
    reference_header = _first_non_empty_row(elements[0].get("table_rows") or [])

    for index, element in enumerate(elements):
        table_rows = list(element.get("table_rows") or [])
        row_html = list(element.get("table_row_html") or [])
        fallback_html.append(str(element.get("table_html", "")))

        element_captions = [str(item).strip() for item in element.get("table_caption") or [] if str(item).strip()]
        element_footnotes = [str(item).strip() for item in element.get("table_footnote") or [] if str(item).strip()]
        captions.extend(item for item in element_captions if item not in captions)
        footnotes.extend(item for item in element_footnotes if item not in footnotes)

        if index > 0 and reference_header and _rows_match(reference_header, _first_non_empty_row(table_rows)):
            table_rows = table_rows[1:] if table_rows else []
            row_html = row_html[1:] if row_html else []

        merged_rows.extend(table_rows)
        merged_row_html.extend(row_html)

    table_text = _collect_table_text(merged_rows, captions=captions, footnotes=footnotes)
    return {
        "text": table_text,
        "captions": captions,
        "footnotes": footnotes,
        "table_html": _build_table_html(merged_row_html, fallback_html),
    }


@dataclass(slots=True)
class StructuredMineruChunker:
    max_chars: int = 1200

    def chunk(self, document: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        section_stack: list[str] = []
        buffer: list[ParsedElement] = []
        elements = document.elements
        index = 0

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            chunks.extend(self._build_text_chunks(document, buffer, section_stack))
            buffer = []

        while index < len(elements):
            element = elements[index]
            kind = element.get("kind")
            text = _normalize_text(str(element.get("text", "")))

            if kind == "heading":
                flush_buffer()
                if text:
                    level = int(element.get("level", 1) or 1)
                    section_stack[:] = section_stack[: max(0, level - 1)]
                    section_stack.append(text)
                index += 1
                continue

            if kind == "table":
                flush_buffer()
                table_elements = [element]
                index += 1
                while index < len(elements):
                    next_element = elements[index]
                    if next_element.get("kind") != "table":
                        break
                    if not _can_merge_tables(table_elements[-1], next_element):
                        break
                    table_elements.append(next_element)
                    index += 1
                chunks.append(self._build_table_chunk(document, table_elements, section_stack))
                continue

            if not text:
                index += 1
                continue

            projected = "\n\n".join([item.get("text", "") for item in buffer] + [text]).strip()
            if buffer and len(projected) > self.max_chars:
                flush_buffer()
            buffer.append(element)
            index += 1

        flush_buffer()

        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk["chunk_id"] = f"{document.doc_id}-chunk-{chunk_index}"
        return chunks

    def _build_text_chunks(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        section_path: list[str],
    ) -> list[ChunkRecord]:
        combined_text = "\n\n".join(
            _normalize_text(str(element.get("text", "")))
            for element in elements
            if _normalize_text(str(element.get("text", "")))
        ).strip()
        if not combined_text:
            return []

        if len(combined_text) <= self.max_chars:
            return [
                self._build_chunk_record(
                    document,
                    elements,
                    combined_text,
                    section_path,
                    chunk_type="paragraph",
                )
            ]

        chunks: list[ChunkRecord] = []
        pending_elements: list[ParsedElement] = []
        pending_texts: list[str] = []
        for element in elements:
            element_text = _normalize_text(str(element.get("text", "")))
            if not element_text:
                continue
            projected = "\n\n".join(pending_texts + [element_text]).strip()
            if pending_elements and len(projected) > self.max_chars:
                chunks.append(
                    self._build_chunk_record(
                        document,
                        pending_elements,
                        "\n\n".join(pending_texts),
                        section_path,
                        chunk_type="paragraph",
                    )
                )
                pending_elements = []
                pending_texts = []

            if len(element_text) > self.max_chars:
                if pending_elements:
                    chunks.append(
                        self._build_chunk_record(
                            document,
                            pending_elements,
                            "\n\n".join(pending_texts),
                            section_path,
                            chunk_type="paragraph",
                        )
                    )
                    pending_elements = []
                    pending_texts = []
                for piece in _split_text_window(element_text, self.max_chars):
                    chunks.append(
                        self._build_chunk_record(
                            document,
                            [element],
                            piece,
                            section_path,
                            chunk_type="paragraph",
                        )
                    )
                continue

            pending_elements.append(element)
            pending_texts.append(element_text)

        if pending_elements:
            chunks.append(
                self._build_chunk_record(
                    document,
                    pending_elements,
                    "\n\n".join(pending_texts),
                    section_path,
                    chunk_type="paragraph",
                )
            )
        return chunks

    def _build_table_chunk(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        section_path: list[str],
    ) -> ChunkRecord:
        merged_table = _merge_table_elements(elements)
        return self._build_chunk_record(
            document,
            elements,
            merged_table["text"],
            section_path,
            chunk_type="table",
            table_html=merged_table["table_html"],
            embedding_text=_build_embedding_text(
                section_path,
                "\n".join(
                    item
                    for item in [*merged_table["captions"], merged_table["text"], *merged_table["footnotes"]]
                    if item
                ),
            ),
        )

    def _build_chunk_record(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        text: str,
        section_path: list[str],
        *,
        chunk_type: str,
        table_html: str = "",
        embedding_text: str | None = None,
    ) -> ChunkRecord:
        page_start, page_end = _page_bounds_from_elements(elements)
        provenance, bbox_refs = _merge_provenance(elements)
        clean_text = _normalize_text(text)
        resolved_embedding_text = embedding_text or _build_embedding_text(section_path, clean_text)
        chunk: ChunkRecord = {
            "chunk_id": "",
            "doc_id": document.doc_id,
            "doc_name": document.doc_name,
            "source_path": document.source_path,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": list(section_path),
            "chunk_type": chunk_type,
            "text": clean_text,
            "embedding_text": resolved_embedding_text,
            "provenance": provenance,
            "bbox_refs": bbox_refs,
            "element_ids": [str(element.get("element_id", "")) for element in elements if element.get("element_id")],
        }
        if table_html:
            chunk["table_html"] = table_html
        return chunk
