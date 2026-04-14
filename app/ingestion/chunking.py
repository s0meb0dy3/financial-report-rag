from dataclasses import dataclass
from typing import Any

from app.ingestion.types import ChunkRecord, ParsedDocument, ParsedElement


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
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

    pieces: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            split_at = normalized.rfind("\n", start, end)
            if split_at <= start:
                split_at = normalized.rfind("。", start, end)
            if split_at > start:
                end = split_at + 1
        piece = normalized[start:end].strip()
        if piece:
            pieces.append(piece)
        start = max(end, start + 1)
    return pieces


@dataclass(slots=True)
class StructuredDoclingChunker:
    max_chars: int = 1200

    def chunk(self, document: ParsedDocument) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        section_stack: list[str] = []
        buffer: list[ParsedElement] = []

        def flush_buffer() -> None:
            nonlocal buffer
            if not buffer:
                return
            chunks.extend(self._build_text_chunks(document, buffer, section_stack))
            buffer = []

        for element in document.elements:
            kind = element.get("kind")
            text = _normalize_text(str(element.get("text", "")))
            if kind == "heading":
                flush_buffer()
                heading = text
                if not heading:
                    continue
                level = max(1, int(element.get("level", 1)))
                section_stack[:] = section_stack[: level - 1]
                section_stack.append(heading)
                continue

            if not text:
                continue

            if kind == "table":
                flush_buffer()
                chunks.extend(self._build_table_chunks(document, element, section_stack))
                continue

            projected = "\n\n".join([item.get("text", "") for item in buffer] + [text]).strip()
            if buffer and len(projected) > self.max_chars:
                flush_buffer()
            buffer.append(element)

        flush_buffer()

        for index, chunk in enumerate(chunks, start=1):
            chunk["chunk_id"] = f"{document.doc_id}-chunk-{index}"
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
            return [self._build_chunk_record(document, elements, combined_text, section_path, chunk_type="paragraph")]

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

    def _build_table_chunks(
        self,
        document: ParsedDocument,
        element: ParsedElement,
        section_path: list[str],
    ) -> list[ChunkRecord]:
        text = _normalize_text(str(element.get("text", "")))
        return [self._build_chunk_record(document, [element], text, section_path, chunk_type="table")]

    def _build_chunk_record(
        self,
        document: ParsedDocument,
        elements: list[ParsedElement],
        text: str,
        section_path: list[str],
        *,
        chunk_type: str,
    ) -> ChunkRecord:
        page_start, page_end = _page_bounds_from_elements(elements)
        provenance, bbox_refs = _merge_provenance(elements)
        return {
            "chunk_id": "",
            "doc_id": document.doc_id,
            "doc_name": document.doc_name,
            "source_path": document.source_path,
            "page": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": list(section_path),
            "chunk_type": chunk_type,
            "text": text,
            "provenance": provenance,
            "bbox_refs": bbox_refs,
            "element_ids": [str(element.get("element_id", "")) for element in elements if element.get("element_id")],
        }
