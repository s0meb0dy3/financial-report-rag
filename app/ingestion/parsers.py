import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import httpx
from pypdf import PdfReader, PdfWriter

from app.ingestion.types import DocumentParser, ParsedDocument, ParsedElement


DEFAULT_MINERU_BASE_URL = "https://mineru.net"
DEFAULT_MINERU_MODEL_VERSION = "vlm"
DEFAULT_MINERU_LANGUAGE = "ch"
DEFAULT_MINERU_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MINERU_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_MINERU_POLL_TIMEOUT_SECONDS = 600.0
DEFAULT_MINERU_MAX_PAGES_PER_REQUEST = 200
_CONTINUATION_CUE_PATTERN = re.compile(r"(续表|continued)", re.IGNORECASE)
_TABLE_ROW_PATTERN = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)


def build_doc_id(pdf_path: Path) -> str:
    digest = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{pdf_path.stem}-{digest}"


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_block_text(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_text_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _normalize_text(value)
        return [text] if text else []
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            text = _normalize_text(value["content"])
            return [text] if text else []
        fragments: list[str] = []
        for child_value in value.values():
            fragments.extend(_extract_text_fragments(child_value))
        return fragments
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_extract_text_fragments(item))
        return fragments
    return []


def _combine_text_fragments(value: Any, *, separator: str = " ") -> str:
    fragments = _extract_text_fragments(value)
    return separator.join(fragment for fragment in fragments if fragment).strip()


def _normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    normalized: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        normalized.append(float(item))
    return normalized


def _infer_heading_level(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 1
    if re.match(r"^第.+?[章节部篇]", stripped):
        return 1
    if re.match(r"^[一二三四五六七八九十]+、", stripped):
        return 2
    if re.match(r"^[(（][一二三四五六七八九十]+[)）]", stripped):
        return 3
    if re.match(r"^\d+[、.]", stripped):
        return 4
    if re.match(r"^[(（]\d+[)）]", stripped):
        return 5
    return 1


class _SimpleTableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, Any]]] = []
        self._current_row: list[dict[str, Any]] = []
        self._current_cell_parts: list[str] = []
        self._current_colspan = 1
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"}:
            attributes = dict(attrs)
            self._current_colspan = _safe_int(attributes.get("colspan")) or 1
            self._current_cell_parts = []
            self._in_cell = True
            return
        if tag == "br" and self._in_cell:
            self._current_cell_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            self._current_row.append(
                {
                    "text": _normalize_block_text("".join(self._current_cell_parts)),
                    "colspan": max(1, self._current_colspan),
                }
            )
            self._current_cell_parts = []
            self._current_colspan = 1
            self._in_cell = False
            return
        if tag == "tr" and self._current_row:
            self.rows.append(self._current_row)
            self._current_row = []


def _parse_html_table(table_html: str) -> tuple[list[list[str]], int]:
    if not table_html.strip():
        return [], 0
    parser = _SimpleTableHTMLParser()
    parser.feed(table_html)
    rows = [[cell["text"] for cell in row] for row in parser.rows]
    column_count = max(
        (sum(int(cell.get("colspan", 1)) for cell in row) for row in parser.rows),
        default=0,
    )
    return rows, column_count


def _extract_table_row_html(table_html: str) -> list[str]:
    return [item.strip() for item in _TABLE_ROW_PATTERN.findall(table_html) if item.strip()]


def _table_rows_to_text(
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
    return _normalize_block_text("\n".join(sections))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_content_paths(artifact_dir: Path) -> tuple[Path | None, Any | None]:
    v2_path = artifact_dir / "content_list_v2.json"
    if v2_path.exists():
        return v2_path, _load_json(v2_path)

    legacy_paths = sorted(artifact_dir.glob("*_content_list.json"))
    for path in legacy_paths:
        return path, _load_json(path)
    return None, None


def _pages_from_legacy_items(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    max_page = 0
    for item in items:
        page_idx = _safe_int(item.get("page_idx"))
        page_number = (page_idx or 0) + 1
        max_page = max(max_page, page_number)
        grouped[page_number].append(item)
    return [grouped.get(page_number, []) for page_number in range(1, max_page + 1)]


def _load_content_pages(artifact_dir: Path) -> tuple[list[list[dict[str, Any]]], Path]:
    path, payload = _candidate_content_paths(artifact_dir)
    if path is None or payload is None:
        raise FileNotFoundError(f"No MinerU content list found in {artifact_dir}")
    if isinstance(payload, list) and (not payload or isinstance(payload[0], list)):
        return payload, path
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return _pages_from_legacy_items(payload), path
    raise ValueError(f"Unsupported MinerU content list payload in {path}")


def _build_provenance(page_number: int, bbox: list[float] | None) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {"page": page_number}
    if bbox is not None:
        entry["bbox"] = bbox
    return [entry]


def _normalize_mineru_v2_block(block: dict[str, Any], *, page_number: int, block_index: int) -> ParsedElement | None:
    block_type = str(block.get("type", "")).strip()
    if block_type in {"page_header", "page_number"}:
        return None

    bbox = _normalize_bbox(block.get("bbox"))
    base_element: ParsedElement = {
        "element_id": f"page-{page_number}-block-{block_index}",
        "page_start": page_number,
        "page_end": page_number,
        "provenance": _build_provenance(page_number, bbox),
    }
    content = block.get("content") or {}

    if block_type == "title":
        text = _combine_text_fragments(content.get("title_content"))
        if not text:
            return None
        return {
            **base_element,
            "kind": "heading",
            "text": text,
            "level": _infer_heading_level(text),
        }

    if block_type == "paragraph":
        text = _combine_text_fragments(content.get("paragraph_content"))
        if not text:
            return None
        return {
            **base_element,
            "kind": "paragraph",
            "text": text,
        }

    if block_type == "list":
        items = content.get("list_items") or []
        item_texts = [
            _combine_text_fragments(item.get("item_content"), separator=" ")
            for item in items
            if isinstance(item, dict)
        ]
        text = _normalize_block_text("\n".join(item for item in item_texts if item))
        if not text:
            return None
        return {
            **base_element,
            "kind": "paragraph",
            "text": text,
        }

    if block_type == "table":
        table_html = str(content.get("html", "")).strip()
        captions = _extract_text_fragments(content.get("table_caption"))
        footnotes = _extract_text_fragments(content.get("table_footnote"))
        rows, column_count = _parse_html_table(table_html)
        text = _table_rows_to_text(rows, captions=captions, footnotes=footnotes)
        if not table_html and not text:
            return None
        continuation_hint = bool(
            _CONTINUATION_CUE_PATTERN.search("\n".join([*captions, *footnotes, text]))
        )
        return {
            **base_element,
            "kind": "table",
            "text": text,
            "table_html": table_html,
            "table_caption": captions,
            "table_footnote": footnotes,
            "table_rows": rows,
            "table_row_html": _extract_table_row_html(table_html),
            "table_column_count": column_count,
            "table_continuation_hint": continuation_hint,
        }

    if block_type == "image":
        caption_texts = _extract_text_fragments(content.get("image_caption"))
        footnote_texts = _extract_text_fragments(content.get("image_footnote"))
        text = _normalize_block_text("\n".join([*caption_texts, *footnote_texts]))
        if not text:
            return None
        return {
            **base_element,
            "kind": "paragraph",
            "text": text,
        }

    text = _combine_text_fragments(content)
    if not text:
        return None
    return {
        **base_element,
        "kind": "paragraph",
        "text": text,
    }


def _normalize_legacy_item(item: dict[str, Any], *, page_number: int, block_index: int) -> ParsedElement | None:
    item_type = str(item.get("type", "")).strip()
    bbox = _normalize_bbox(item.get("bbox"))
    base_element: ParsedElement = {
        "element_id": f"page-{page_number}-block-{block_index}",
        "page_start": page_number,
        "page_end": page_number,
        "provenance": _build_provenance(page_number, bbox),
    }

    if item_type == "title":
        text = _normalize_text(str(item.get("text", "")))
        if not text:
            return None
        return {
            **base_element,
            "kind": "heading",
            "text": text,
            "level": _infer_heading_level(text),
        }

    if item_type == "table":
        table_html = str(item.get("html") or item.get("table_html") or "").strip()
        rows, column_count = _parse_html_table(table_html)
        text = _table_rows_to_text(rows)
        if not table_html and not text:
            text = _normalize_text(str(item.get("text", "")))
        if not table_html and not text:
            return None
        return {
            **base_element,
            "kind": "table",
            "text": text,
            "table_html": table_html,
            "table_caption": [],
            "table_footnote": [],
            "table_rows": rows,
            "table_row_html": _extract_table_row_html(table_html),
            "table_column_count": column_count,
            "table_continuation_hint": bool(_CONTINUATION_CUE_PATTERN.search(text)),
        }

    if item_type in {"page_header", "page_number", "image"}:
        return None

    text = _normalize_text(str(item.get("text", "")))
    if not text:
        return None
    return {
        **base_element,
        "kind": "paragraph",
        "text": text,
    }


class MineruPdfParser:
    @classmethod
    def from_env(
        cls,
        artifact_root: Path,
        *,
        force_parse: bool = False,
        client: httpx.Client | None = None,
    ) -> "MineruPdfParser":
        return cls(
            artifact_root=artifact_root,
            api_token=os.environ.get("MINERU_API_TOKEN", ""),
            base_url=os.environ.get("MINERU_BASE_URL", DEFAULT_MINERU_BASE_URL),
            model_version=os.environ.get("MINERU_MODEL_VERSION", DEFAULT_MINERU_MODEL_VERSION),
            language=os.environ.get("MINERU_LANGUAGE", DEFAULT_MINERU_LANGUAGE),
            request_timeout=float(
                os.environ.get("MINERU_REQUEST_TIMEOUT_SECONDS", DEFAULT_MINERU_REQUEST_TIMEOUT_SECONDS)
            ),
            poll_interval_seconds=float(
                os.environ.get("MINERU_POLL_INTERVAL_SECONDS", DEFAULT_MINERU_POLL_INTERVAL_SECONDS)
            ),
            poll_timeout_seconds=float(
                os.environ.get("MINERU_POLL_TIMEOUT_SECONDS", DEFAULT_MINERU_POLL_TIMEOUT_SECONDS)
            ),
            max_pages_per_request=int(
                os.environ.get("MINERU_MAX_PAGES_PER_REQUEST", DEFAULT_MINERU_MAX_PAGES_PER_REQUEST)
            ),
            force_parse=force_parse,
            client=client,
        )

    def __init__(
        self,
        *,
        artifact_root: Path,
        api_token: str = "",
        base_url: str = DEFAULT_MINERU_BASE_URL,
        model_version: str = DEFAULT_MINERU_MODEL_VERSION,
        language: str = DEFAULT_MINERU_LANGUAGE,
        request_timeout: float = DEFAULT_MINERU_REQUEST_TIMEOUT_SECONDS,
        poll_interval_seconds: float = DEFAULT_MINERU_POLL_INTERVAL_SECONDS,
        poll_timeout_seconds: float = DEFAULT_MINERU_POLL_TIMEOUT_SECONDS,
        max_pages_per_request: int = DEFAULT_MINERU_MAX_PAGES_PER_REQUEST,
        force_parse: bool = False,
        client: httpx.Client | None = None,
    ):
        self.artifact_root = Path(artifact_root)
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.language = language
        self.request_timeout = request_timeout
        self.poll_interval_seconds = max(1.0, poll_interval_seconds)
        self.poll_timeout_seconds = max(self.poll_interval_seconds, poll_timeout_seconds)
        self.max_pages_per_request = max(1, int(max_pages_per_request))
        self.force_parse = force_parse
        self._client = client
        self._owns_client = client is None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.request_timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        if not self.api_token:
            raise ValueError("MINERU_API_TOKEN is not set")
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

    def _artifact_dir_for_doc(self, doc_id: str) -> Path:
        return self.artifact_root / doc_id

    def _manifest_path(self, artifact_dir: Path) -> Path:
        return artifact_dir / "manifest.json"

    def _build_manifest(self, pdf_path: Path, doc_id: str) -> dict[str, Any]:
        stat = pdf_path.stat()
        return {
            "parser": "mineru-precision-v1",
            "doc_id": doc_id,
            "source_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "base_url": self.base_url,
            "model_version": self.model_version,
            "language": self.language,
        }

    def _cache_matches(self, artifact_dir: Path, expected_manifest: dict[str, Any]) -> bool:
        manifest_path = self._manifest_path(artifact_dir)
        if not manifest_path.exists():
            return False
        try:
            existing_manifest = _load_json(manifest_path)
        except (json.JSONDecodeError, OSError):
            return False
        for key, value in expected_manifest.items():
            if existing_manifest.get(key) != value:
                return False
        try:
            _load_content_pages(artifact_dir)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def _pdf_page_count(self, pdf_path: Path) -> int | None:
        try:
            return len(PdfReader(str(pdf_path)).pages)
        except Exception:
            return None

    def _page_ranges(self, page_count: int) -> list[tuple[int, int]]:
        return [
            (start, min(start + self.max_pages_per_request - 1, page_count))
            for start in range(1, page_count + 1, self.max_pages_per_request)
        ]

    def _part_doc_id(self, doc_id: str, part_index: int) -> str:
        return f"{doc_id}-part-{part_index:03d}"

    def _part_pdf_name(self, part_index: int, page_start: int, page_end: int) -> str:
        return f"part-{part_index:03d}-pages-{page_start:03d}-{page_end:03d}.pdf"

    def _build_split_manifest(
        self,
        pdf_path: Path,
        doc_id: str,
        artifact_dir: Path,
        page_count: int,
        page_ranges: list[tuple[int, int]],
    ) -> dict[str, Any]:
        split_dir = artifact_dir / "split_pdfs"
        parts_dir = artifact_dir / "parts"
        parts = [
            {
                "part_index": part_index,
                "doc_id": self._part_doc_id(doc_id, part_index),
                "page_start": page_start,
                "page_end": page_end,
                "pdf_path": str(split_dir / self._part_pdf_name(part_index, page_start, page_end)),
                "artifact_dir": str(parts_dir / f"part-{part_index:03d}"),
            }
            for part_index, (page_start, page_end) in enumerate(page_ranges, start=1)
        ]
        return {
            **self._build_manifest(pdf_path, doc_id),
            "split": True,
            "page_count": page_count,
            "part_count": len(page_ranges),
            "max_pages_per_request": self.max_pages_per_request,
            "parts": parts,
        }

    def _split_cache_matches(self, artifact_dir: Path, expected_manifest: dict[str, Any]) -> bool:
        manifest_path = self._manifest_path(artifact_dir)
        if not manifest_path.exists():
            return False
        try:
            existing_manifest = _load_json(manifest_path)
        except (json.JSONDecodeError, OSError):
            return False
        for key, value in expected_manifest.items():
            if existing_manifest.get(key) != value:
                return False
        return all(Path(part["pdf_path"]).exists() for part in expected_manifest["parts"])

    def _write_split_pdfs(
        self,
        pdf_path: Path,
        page_ranges: list[tuple[int, int]],
        split_dir: Path,
    ) -> None:
        reader = PdfReader(str(pdf_path))
        split_dir.mkdir(parents=True, exist_ok=True)
        for part_index, (page_start, page_end) in enumerate(page_ranges, start=1):
            writer = PdfWriter()
            for page_index in range(page_start - 1, page_end):
                writer.add_page(reader.pages[page_index])
            part_path = split_dir / self._part_pdf_name(part_index, page_start, page_end)
            with part_path.open("wb") as output_file:
                writer.write(output_file)

    def _prepare_split_artifacts(
        self,
        pdf_path: Path,
        doc_id: str,
        artifact_dir: Path,
        page_count: int,
        page_ranges: list[tuple[int, int]],
    ) -> dict[str, Any]:
        manifest = self._build_split_manifest(pdf_path, doc_id, artifact_dir, page_count, page_ranges)
        if self.force_parse or not self._split_cache_matches(artifact_dir, manifest):
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            self._write_split_pdfs(pdf_path, page_ranges, artifact_dir / "split_pdfs")
            self._manifest_path(artifact_dir).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return manifest

    def _request_upload_target(self, pdf_path: Path, doc_id: str) -> tuple[str, str]:
        response = self.client.post(
            f"{self.base_url}/api/v4/file-urls/batch",
            headers=self._headers(),
            json={
                "files": [{"name": pdf_path.name, "data_id": doc_id}],
                "language": self.language,
                "enable_table": True,
                "enable_formula": True,
                "model_version": self.model_version,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise ValueError(payload.get("msg") or "MinerU upload target request failed")
        data = payload.get("data") or {}
        batch_id = str(data.get("batch_id", "")).strip()
        file_urls = data.get("file_urls") or []
        file_url = str(file_urls[0]).strip() if file_urls else ""
        if not batch_id or not file_url:
            raise ValueError("MinerU upload target response is missing batch_id or file_urls")
        return batch_id, file_url

    def _upload_file(self, pdf_path: Path, file_url: str) -> None:
        with pdf_path.open("rb") as file_obj:
            response = self.client.put(file_url, content=file_obj)
        response.raise_for_status()

    def _sleep(self) -> None:
        import time

        time.sleep(self.poll_interval_seconds)

    def _poll_until_done(self, batch_id: str) -> dict[str, Any]:
        elapsed = 0.0
        while elapsed <= self.poll_timeout_seconds:
            response = self.client.get(
                f"{self.base_url}/api/v4/extract-results/batch/{batch_id}",
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Accept": "*/*",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise ValueError(payload.get("msg") or "MinerU result polling failed")
            result = (payload.get("data", {}).get("extract_result") or [{}])[0]
            state = str(result.get("state", "")).strip()
            if state == "done":
                return result
            if state == "failed":
                raise ValueError(result.get("err_msg") or "MinerU parse failed")
            elapsed += self.poll_interval_seconds
            if elapsed > self.poll_timeout_seconds:
                break
            self._sleep()
        raise TimeoutError(f"MinerU parse polling timed out for batch_id={batch_id}")

    def _download_zip(self, result_zip_url: str, zip_path: Path) -> None:
        with self.client.stream("GET", result_zip_url, follow_redirects=True) as response:
            response.raise_for_status()
            with zip_path.open("wb") as output_file:
                for chunk in response.iter_bytes():
                    output_file.write(chunk)

    def _fetch_and_cache(
        self,
        pdf_path: Path,
        doc_id: str,
        artifact_dir: Path,
        manifest: dict[str, Any],
    ) -> None:
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        batch_id, file_url = self._request_upload_target(pdf_path, doc_id)
        self._upload_file(pdf_path, file_url)
        result = self._poll_until_done(batch_id)
        result_zip_url = str(result.get("full_zip_url", "")).strip()
        if not result_zip_url:
            raise ValueError("MinerU parse result is missing full_zip_url")

        zip_path = artifact_dir / "result.zip"
        self._download_zip(result_zip_url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(artifact_dir)

        manifest_payload = {
            **manifest,
            "batch_id": batch_id,
            "result_zip_url": result_zip_url,
        }
        self._manifest_path(artifact_dir).write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _collect_elements(self, pages: list[list[dict[str, Any]]], *, legacy: bool) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        for page_index, blocks in enumerate(pages, start=1):
            for block_index, block in enumerate(blocks, start=1):
                if not isinstance(block, dict):
                    continue
                normalized = (
                    _normalize_legacy_item(block, page_number=page_index, block_index=block_index)
                    if legacy
                    else _normalize_mineru_v2_block(block, page_number=page_index, block_index=block_index)
                )
                if normalized is not None:
                    elements.append(normalized)
        return elements

    def _build_parsed_document(
        self,
        *,
        pdf_path: Path,
        doc_id: str,
        artifact_dir: Path,
        page_count: int | None,
    ) -> ParsedDocument:
        content_pages, content_path = _load_content_pages(artifact_dir)
        markdown_path = artifact_dir / "full.md"
        markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else None
        elements = self._collect_elements(
            content_pages,
            legacy=content_path.name != "content_list_v2.json",
        )

        return ParsedDocument(
            doc_id=doc_id,
            doc_name=pdf_path.name,
            source_path=str(pdf_path),
            raw_doc={
                "parser": "mineru",
                "artifact_dir": str(artifact_dir),
                "manifest_path": str(self._manifest_path(artifact_dir)),
                "content_list_path": str(content_path),
                "result_zip_path": str(artifact_dir / "result.zip"),
                "split": False,
                "page_count": page_count,
                "part_count": 1,
                "max_pages_per_request": self.max_pages_per_request,
            },
            markdown=markdown,
            elements=elements,
            page_map={
                page_number: {"block_count": len(page_blocks)}
                for page_number, page_blocks in enumerate(content_pages, start=1)
            },
        )

    def _offset_element_pages(
        self,
        element: ParsedElement,
        *,
        page_offset: int,
        part_index: int,
        element_index: int,
    ) -> ParsedElement:
        adjusted = deepcopy(element)
        element_id = str(adjusted.get("element_id", "")).strip()
        adjusted["element_id"] = (
            f"part-{part_index:03d}-{element_id}"
            if element_id
            else f"part-{part_index:03d}-element-{element_index:06d}"
        )
        for key in ("page_start", "page_end"):
            page_number = _safe_int(adjusted.get(key))
            if page_number is not None:
                adjusted[key] = page_number + page_offset
        provenance = adjusted.get("provenance")
        if isinstance(provenance, list):
            for item in provenance:
                if not isinstance(item, dict):
                    continue
                for key in ("page", "page_no"):
                    page_number = _safe_int(item.get(key))
                    if page_number is not None:
                        item[key] = page_number + page_offset
        return adjusted

    def _load_part_parse_result(
        self,
        *,
        part_pdf_path: Path,
        part_doc_id: str,
        part_artifact_dir: Path,
    ) -> tuple[list[list[dict[str, Any]]], Path, str | None, list[ParsedElement]]:
        manifest = self._build_manifest(part_pdf_path, part_doc_id)
        if self.force_parse or not self._cache_matches(part_artifact_dir, manifest):
            self._fetch_and_cache(part_pdf_path, part_doc_id, part_artifact_dir, manifest)

        content_pages, content_path = _load_content_pages(part_artifact_dir)
        markdown_path = part_artifact_dir / "full.md"
        markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else None
        elements = self._collect_elements(
            content_pages,
            legacy=content_path.name != "content_list_v2.json",
        )
        return content_pages, content_path, markdown, elements

    def _parse_split_document(self, pdf_path: Path, doc_id: str, page_count: int) -> ParsedDocument:
        artifact_dir = self._artifact_dir_for_doc(doc_id)
        page_ranges = self._page_ranges(page_count)
        manifest = self._prepare_split_artifacts(pdf_path, doc_id, artifact_dir, page_count, page_ranges)

        combined_elements: list[ParsedElement] = []
        combined_page_map: dict[int, dict[str, Any]] = {}
        markdown_parts: list[str] = []
        part_records: list[dict[str, Any]] = []

        for part in manifest["parts"]:
            part_index = int(part["part_index"])
            page_start = int(part["page_start"])
            page_end = int(part["page_end"])
            page_offset = page_start - 1
            part_pdf_path = Path(part["pdf_path"])
            part_artifact_dir = Path(part["artifact_dir"])
            content_pages, content_path, markdown, elements = self._load_part_parse_result(
                part_pdf_path=part_pdf_path,
                part_doc_id=str(part["doc_id"]),
                part_artifact_dir=part_artifact_dir,
            )

            if markdown:
                markdown_parts.append(markdown)
            for local_page_number, blocks in enumerate(content_pages, start=1):
                global_page_number = page_offset + local_page_number
                combined_page_map[global_page_number] = {
                    "block_count": len(blocks),
                    "part_index": part_index,
                }
            for element_index, element in enumerate(elements, start=1):
                combined_elements.append(
                    self._offset_element_pages(
                        element,
                        page_offset=page_offset,
                        part_index=part_index,
                        element_index=element_index,
                    )
                )

            part_records.append(
                {
                    **part,
                    "content_list_path": str(content_path),
                    "manifest_path": str(self._manifest_path(part_artifact_dir)),
                    "result_zip_path": str(part_artifact_dir / "result.zip"),
                }
            )

        raw_doc = {
            "parser": "mineru",
            "artifact_dir": str(artifact_dir),
            "manifest_path": str(self._manifest_path(artifact_dir)),
            "split": True,
            "page_count": page_count,
            "part_count": len(page_ranges),
            "max_pages_per_request": self.max_pages_per_request,
            "parts": part_records,
        }
        return ParsedDocument(
            doc_id=doc_id,
            doc_name=pdf_path.name,
            source_path=str(pdf_path),
            raw_doc=raw_doc,
            markdown="\n\n".join(markdown_parts) if markdown_parts else None,
            elements=combined_elements,
            page_map=combined_page_map,
        )

    def parse(self, pdf_path: Path) -> ParsedDocument:
        path = Path(pdf_path).resolve()
        doc_id = build_doc_id(path)
        page_count = self._pdf_page_count(path)
        if page_count is not None and page_count > self.max_pages_per_request:
            return self._parse_split_document(path, doc_id, page_count)

        manifest = self._build_manifest(path, doc_id)
        artifact_dir = self._artifact_dir_for_doc(doc_id)

        if self.force_parse or not self._cache_matches(artifact_dir, manifest):
            self._fetch_and_cache(path, doc_id, artifact_dir, manifest)

        return self._build_parsed_document(
            pdf_path=path,
            doc_id=doc_id,
            artifact_dir=artifact_dir,
            page_count=page_count,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        self.close()


Parser = MineruPdfParser
