import hashlib
from pathlib import Path
from typing import Any

from app.ingestion.types import DocumentParser, ParsedDocument, ParsedElement


def build_doc_id(pdf_path: Path) -> str:
    digest = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{pdf_path.stem}-{digest}"


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _extract_pages_from_provenance(provenance: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    page_numbers = []
    for item in provenance:
        page_no = item.get("page") or item.get("page_no")
        if isinstance(page_no, int):
            page_numbers.append(page_no)
    if not page_numbers:
        return None, None
    return min(page_numbers), max(page_numbers)


def _table_matrix_from_data(table_data: dict[str, Any]) -> list[list[str]]:
    grid = table_data.get("grid") or []
    rows: list[list[str]] = []
    for row in grid:
        cells = []
        for cell in row or []:
            if not isinstance(cell, dict):
                continue
            text = str(cell.get("text", "")).replace("\n", " ").strip()
            cells.append(text)
        if cells:
            rows.append(cells)
    if not rows:
        cells = table_data.get("table_cells") or []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            row_index = int(cell.get("start_row_offset_idx", 0))
            while len(rows) <= row_index:
                rows.append([])
            rows[row_index].append(str(cell.get("text", "")).replace("\n", " ").strip())
    return rows


def _markdown_table_from_data(table_data: dict[str, Any]) -> str:
    rows = _table_matrix_from_data(table_data)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    separator = ["---"] * width
    body = normalized_rows[1:] or [[]]
    markdown_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)


def _resolve_texts(refs: list[Any], ref_map: dict[str, dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        ref_value = ref.get("$ref")
        if not isinstance(ref_value, str):
            continue
        item = ref_map.get(ref_value)
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def _looks_like_table_note(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return True
    note_markers = ("单位：", "币种：", "注：", "说明：", "金额单位：")
    return compact.startswith(note_markers)


def _normalize_docling_item(
    item: dict[str, Any],
    ref_map: dict[str, dict[str, Any]],
) -> ParsedElement | None:
    ref = str(item.get("self_ref", ""))
    element_id = ref.split("/")[-1] if ref else ""
    provenance = item.get("prov") or []
    if not isinstance(provenance, list):
        provenance = []
    normalized_prov = [entry for entry in provenance if isinstance(entry, dict)]
    page_start, page_end = _extract_pages_from_provenance(normalized_prov)

    label = str(item.get("label", ""))
    text = str(item.get("text", "")).strip()
    kind = "paragraph"
    level = None

    if label == "section_header":
        kind = "heading"
        level = _infer_heading_level(text)
    elif label in {"page_header", "page_footer"} or item.get("content_layer") == "furniture":
        return None

    if "data" in item:
        kind = "table"
        table_data = item.get("data") or {}
        matrix = _table_matrix_from_data(table_data)
        text = _markdown_table_from_data(table_data)
        caption_texts = _resolve_texts(item.get("captions") or [], ref_map)
        footnote_texts = _resolve_texts(item.get("footnotes") or [], ref_map)
        title = next((caption for caption in caption_texts if not _looks_like_table_note(caption)), "")
        note_texts = footnote_texts + [
            caption
            for caption in caption_texts
            if caption != title
        ]

    if kind == "table":
        if not matrix and not text:
            return None
    elif not text:
        return None

    element: ParsedElement = {
        "element_id": element_id,
        "kind": kind,
        "text": text,
        "page_start": page_start,
        "page_end": page_end,
        "provenance": [{"page": entry.get("page_no"), "bbox": entry.get("bbox")} for entry in normalized_prov],
    }
    if level is not None:
        element["level"] = level
    if kind == "table":
        element["matrix"] = matrix
        element["title"] = title
        element["footnotes_text"] = "\n".join(note_texts).strip()
    return element


def _infer_heading_level(text: str) -> int:
    stripped = text.strip()
    if stripped.startswith("第") and "节" in stripped[:8]:
        return 1
    if stripped[:2] in {"一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、", "十、"}:
        return 2
    if stripped.startswith("（") and "）" in stripped[:5]:
        return 3
    return 1


def _build_page_map(raw_doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages = raw_doc.get("pages") or {}
    if isinstance(pages, list):
        return {
            index + 1: page
            for index, page in enumerate(pages)
            if isinstance(page, dict)
        }
    if isinstance(pages, dict):
        page_map: dict[int, dict[str, Any]] = {}
        for key, value in pages.items():
            try:
                page_map[int(key)] = value
            except (TypeError, ValueError):
                continue
        return page_map
    return {}


def _resolve_ref(ref: Any, ref_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(ref, dict):
        return None
    ref_value = ref.get("$ref")
    if not isinstance(ref_value, str):
        return None
    return ref_map.get(ref_value)


def _collect_body_elements(raw_doc: dict[str, Any]) -> list[ParsedElement]:
    ref_map: dict[str, dict[str, Any]] = {}
    for collection_name in ("texts", "tables", "groups"):
        for item in raw_doc.get(collection_name) or []:
            if isinstance(item, dict) and isinstance(item.get("self_ref"), str):
                ref_map[item["self_ref"]] = item

    body = raw_doc.get("body") or {}
    body_children = body.get("children") or []
    elements: list[ParsedElement] = []

    def walk(items: list[Any]) -> None:
        for child in items:
            resolved = _resolve_ref(child, ref_map)
            if resolved is None:
                continue
            if "data" in resolved or "text" in resolved:
                normalized = _normalize_docling_item(resolved, ref_map)
                if normalized is not None:
                    elements.append(normalized)
                continue
            if isinstance(resolved.get("children"), list):
                walk(resolved["children"])

    walk(body_children)
    return elements


class DoclingPdfParser:
    def __init__(self, converter: Any | None = None):
        self._converter = converter

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def parse(self, pdf_path: Path) -> ParsedDocument:
        path = Path(pdf_path).resolve()
        converter = self._get_converter()
        result = converter.convert(path)
        document = result.document
        raw_doc = document.export_to_dict()
        try:
            markdown = document.export_to_markdown()
        except Exception:
            markdown = None

        return ParsedDocument(
            doc_id=build_doc_id(path),
            doc_name=path.name,
            source_path=str(path),
            raw_doc=raw_doc,
            markdown=markdown,
            elements=_collect_body_elements(raw_doc),
            page_map=_build_page_map(raw_doc),
        )
