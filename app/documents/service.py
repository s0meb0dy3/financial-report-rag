import json
import http.client
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Any


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_MINERU_DIR = Path("data/processed/mineru")
MINERU_API_BASE_URL = "https://mineru.net/api/v4"
MINERU_MAX_PAGES_PER_FILE = 200


class DocumentServiceError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    name: str
    page_count: int
    pdf_path: Path
    artifact_dir: Path
    parsed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "page_count": self.page_count,
            "parsed": self.parsed,
        }


@dataclass(frozen=True)
class DocumentPage:
    doc_id: str
    doc_name: str
    page: int
    text: str
    blocks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page": self.page,
            "text": self.text,
            "blocks": self.blocks,
        }


class DocumentService:
    """Reads local PDFs and MinerU page-level parse artifacts.

    The chat layer only needs document metadata and page text. Keeping that file
    parsing here avoids rebuilding the old RAG/indexing pipeline.
    """

    def __init__(
        self,
        *,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        mineru_dir: str | Path = DEFAULT_MINERU_DIR,
        mineru_api_key: str = "",
    ):
        self.raw_dir = Path(raw_dir)
        self.mineru_dir = Path(mineru_dir)
        self.mineru_api_key = mineru_api_key

    def get_toc(self, doc_id: str) -> dict[str, Any]:
        """Return the table of contents (bookmarks) from a PDF.

        Uses PyMuPDF to read embedded PDF outlines. Returns entries with both
        physical page numbers (for read_pdf_page) and logical page labels.
        """
        import pymupdf

        doc = self.get_document(doc_id)
        pdf_path = self.get_pdf_path(doc_id)

        pdf = pymupdf.open(str(pdf_path))
        try:
            toc = pdf.get_toc(simple=True)
            page_labels = _build_page_label_map(pdf)
        finally:
            pdf.close()

        entries: list[dict[str, Any]] = []
        for level, title, physical_page in toc:
            if not title:
                continue
            logical = page_labels.get(physical_page)
            entries.append({
                "level": level,
                "title": title.strip(),
                "page": physical_page,
                "page_label": logical,
            })

        summary = f"共 {doc.page_count} 页"
        first_label = page_labels.get(1)
        if first_label and first_label != "1":
            summary += f"，逻辑页码从 {first_label} 开始（封面等前言页不计入正文页码）"

        return {
            "doc_id": doc.id,
            "doc_name": doc.name,
            "page_count": doc.page_count,
            "summary": summary,
            "entries": entries,
        }

    def list_documents(self) -> list[DocumentInfo]:
        docs: list[DocumentInfo] = []
        if self.mineru_dir.exists():
            for manifest_path in sorted(self.mineru_dir.glob("*/manifest.json")):
                try:
                    docs.append(self._document_from_manifest(manifest_path))
                except DocumentServiceError:
                    continue
        parsed_pdf_paths = {doc.pdf_path.resolve() for doc in docs if doc.pdf_path.exists()}
        for pdf_path in self._uploaded_pdf_paths():
            if pdf_path.resolve() in parsed_pdf_paths:
                continue
            docs.append(
                DocumentInfo(
                    id=_uploaded_doc_id(pdf_path),
                    name=pdf_path.name,
                    page_count=_pdf_page_count(pdf_path),
                    pdf_path=pdf_path,
                    artifact_dir=self.mineru_dir / _uploaded_doc_id(pdf_path),
                    parsed=False,
                )
            )
        return sorted(docs, key=lambda item: item.name)

    def get_document(self, doc_id: str) -> DocumentInfo:
        for doc in self.list_documents():
            if doc.id == doc_id:
                return doc
        raise DocumentServiceError(f"Unknown document: {doc_id}")

    def get_pdf_path(self, doc_id: str) -> Path:
        doc = self.get_document(doc_id)
        if not doc.pdf_path.exists():
            raise DocumentServiceError(f"PDF file is missing for document: {doc_id}")
        return doc.pdf_path

    def save_upload(self, file_name: str, content: bytes) -> DocumentInfo:
        safe_name = _safe_pdf_name(file_name)
        if not content:
            raise DocumentServiceError("Uploaded PDF is empty")
        page_count = _uploaded_pdf_page_count(content)
        upload_dir = self.raw_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / safe_name
        if target.exists():
            stem = target.stem
            suffix = target.suffix
            counter = 2
            while target.exists():
                target = upload_dir / f"{stem}-{counter}{suffix}"
                counter += 1
        target.write_bytes(content)
        doc_id = _uploaded_doc_id(target)
        artifact_dir = self.mineru_dir / doc_id
        try:
            if self.mineru_api_key:
                _parse_with_mineru_api(self.mineru_api_key, target, content, page_count, artifact_dir)
            else:
                _write_local_parse(artifact_dir, doc_id, target, _uploaded_pdf_pages(content))
        except DocumentServiceError:
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            if target.exists():
                target.unlink()
            raise
        _read_json.cache_clear()
        return self.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        doc = self.get_document(doc_id)
        if not doc.id.startswith("upload-"):
            return False
        removed = False
        if _is_relative_to(doc.artifact_dir.resolve(), self.mineru_dir.resolve()) and doc.artifact_dir.exists():
            shutil.rmtree(doc.artifact_dir)
            removed = True
        allowed_pdf_roots = [self.raw_dir.resolve(), (self.raw_dir / "uploads").resolve()]
        pdf_path = doc.pdf_path.resolve()
        if doc.pdf_path.exists() and any(_is_relative_to(pdf_path, root) for root in allowed_pdf_roots):
            doc.pdf_path.unlink()
            removed = True
        _read_json.cache_clear()
        return removed

    def read_page(self, doc_id: str, page: int) -> DocumentPage:
        doc = self.get_document(doc_id)
        if not doc.parsed:
            raise DocumentServiceError(f"Document is not parsed yet: {doc_id}")
        if page < 1 or page > doc.page_count:
            raise DocumentServiceError(f"page must be between 1 and {doc.page_count}")

        manifest = _read_json(doc.artifact_dir / "manifest.json")
        page_items = self._read_content_page(doc.artifact_dir, manifest, page)
        blocks = [_normalize_block(item) for item in page_items if isinstance(item, dict)]
        text = "\n".join(block["text"] for block in blocks if block["text"]).strip()
        return DocumentPage(
            doc_id=doc.id,
            doc_name=doc.name,
            page=page,
            text=text,
            blocks=blocks,
        )

    def _document_from_manifest(self, manifest_path: Path) -> DocumentInfo:
        manifest = _read_json(manifest_path)
        doc_id = str(manifest.get("doc_id") or manifest_path.parent.name)
        name = str(manifest.get("file_name") or doc_id)
        page_count = _page_count(manifest, manifest_path.parent)
        pdf_path = self._resolve_pdf_path(manifest, name, manifest_path.parent)
        if page_count <= 0:
            raise DocumentServiceError(f"Document has no parsed pages: {doc_id}")
        return DocumentInfo(
            id=doc_id,
            name=name,
            page_count=page_count,
            pdf_path=pdf_path,
            artifact_dir=manifest_path.parent,
        )

    def _resolve_pdf_path(self, manifest: dict[str, Any], file_name: str, artifact_dir: Path) -> Path:
        candidates: list[Path] = []
        source_path = manifest.get("source_path")
        if isinstance(source_path, str) and source_path:
            candidates.append(Path(source_path))
        candidates.extend(
            [
                self.raw_dir / file_name,
                self.raw_dir / "uploads" / file_name,
                artifact_dir / f"{artifact_dir.name}_origin.pdf",
            ]
        )
        candidates.extend(sorted(artifact_dir.glob("*_origin.pdf")))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        basename = Path(file_name).name
        for root in (self.raw_dir, self.raw_dir / "uploads"):
            if not root.exists():
                continue
            for candidate in root.glob("*.pdf"):
                if candidate.name == basename or basename in candidate.name or candidate.name in basename:
                    return candidate
        return candidates[0] if candidates else self.raw_dir / file_name

    def _read_content_page(
        self,
        artifact_dir: Path,
        manifest: dict[str, Any],
        page: int,
    ) -> list[dict[str, Any]]:
        if manifest.get("split") and isinstance(manifest.get("parts"), list):
            for part in manifest["parts"]:
                if not isinstance(part, dict):
                    continue
                start = int(part.get("page_start") or 0)
                end = int(part.get("page_end") or 0)
                if start <= page <= end:
                    part_dir = Path(str(part.get("artifact_dir") or ""))
                    if not part_dir.exists():
                        part_dir = artifact_dir / "parts" / f"part-{int(part.get('part_index', 1)):03d}"
                    return _read_page_from_content_list(part_dir / "content_list_v2.json", page - start + 1)
        return _read_page_from_content_list(artifact_dir / "content_list_v2.json", page)

    def _uploaded_pdf_paths(self) -> list[Path]:
        paths: list[Path] = []
        for root in (self.raw_dir, self.raw_dir / "uploads"):
            if root.exists():
                paths.extend(sorted(root.glob("*.pdf")))
        return paths


def _safe_pdf_name(file_name: str) -> str:
    name = Path(file_name).name.strip() or "document.pdf"
    if not name.lower().endswith(".pdf"):
        raise DocumentServiceError("Only PDF files are supported")
    safe = re.sub(r"[^\w\-.一-鿿]+", "-", name).strip(".-")
    return safe if safe.lower().endswith(".pdf") else f"{safe or 'document'}.pdf"


def _uploaded_doc_id(pdf_path: Path) -> str:
    return f"upload-{re.sub(r'[^a-zA-Z0-9_-]+', '-', pdf_path.stem).strip('-').lower() or 'document'}"


def _uploaded_pdf_page_count(content: bytes) -> int:
    try:
        import pymupdf

        pdf = pymupdf.open(stream=content, filetype="pdf")
        try:
            page_count = int(pdf.page_count)
        finally:
            pdf.close()
    except Exception as exc:
        raise DocumentServiceError("Uploaded PDF is invalid") from exc
    if page_count <= 0:
        raise DocumentServiceError("Uploaded PDF has no pages")
    return page_count


def _uploaded_pdf_pages(content: bytes) -> list[list[dict[str, Any]]]:
    try:
        import pymupdf

        pdf = pymupdf.open(stream=content, filetype="pdf")
        try:
            page_count = int(pdf.page_count)
            pages = [_page_to_block_list(pdf[index].get_text("text")) for index in range(page_count)]
        finally:
            pdf.close()
    except Exception as exc:
        raise DocumentServiceError("Uploaded PDF is invalid") from exc
    if page_count <= 0:
        raise DocumentServiceError("Uploaded PDF has no pages")
    return pages


def _write_local_parse(artifact_dir: Path, doc_id: str, pdf_path: Path, pages: list[list[dict[str, Any]]]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        artifact_dir / "manifest.json",
        {
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "source_path": str(pdf_path),
            "page_count": len(pages),
            "parser": "pymupdf",
        },
    )
    _write_json(artifact_dir / "content_list_v2.json", pages)


def _parse_with_mineru_api(
    api_key: str,
    pdf_path: Path,
    content: bytes,
    page_count: int,
    artifact_dir: Path,
) -> None:
    doc_id = _uploaded_doc_id(pdf_path)
    ranges = _mineru_page_ranges(page_count)
    files = [
        {
            "name": f"{pdf_path.stem}-part-{index:03d}{pdf_path.suffix}",
            "is_ocr": True,
            "data_id": f"{doc_id}-part-{index:03d}",
            "page_ranges": f"{start}-{end}",
        }
        for index, (start, end) in enumerate(ranges, start=1)
    ]
    payload = {
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
        "files": files,
    }
    applied = _mineru_json("POST", f"{MINERU_API_BASE_URL}/file-urls/batch", api_key, payload)
    data = applied.get("data") if isinstance(applied, dict) else None
    if not isinstance(data, dict):
        raise DocumentServiceError("MinerU did not return upload data")
    batch_id = str(data.get("batch_id") or "")
    upload_urls = data.get("file_urls")
    if not batch_id or not isinstance(upload_urls, list):
        raise DocumentServiceError("MinerU did not return upload URLs")
    for item in upload_urls:
        upload_url = _mineru_upload_url(item)
        if not upload_url:
            raise DocumentServiceError("MinerU returned an invalid upload URL")
        _mineru_put_bytes(upload_url, content)

    results = _wait_for_mineru_results(api_key, batch_id, len(files))
    if len(ranges) == 1:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _download_mineru_zip(results[0], artifact_dir)
        _normalize_content_list(artifact_dir)
        _write_json(
            artifact_dir / "manifest.json",
            {
                "doc_id": doc_id,
                "file_name": pdf_path.name,
                "source_path": str(pdf_path),
                "page_count": page_count,
                "parser": "mineru_api_precise",
                "batch_id": batch_id,
            },
        )
        return

    parts = []
    for index, ((start, end), result) in enumerate(zip(ranges, results, strict=True), start=1):
        part_dir = artifact_dir / "parts" / f"part-{index:03d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        _download_mineru_zip(result, part_dir)
        _normalize_content_list(part_dir)
        parts.append(
            {
                "part_index": index,
                "page_start": start,
                "page_end": end,
                "artifact_dir": str(part_dir),
            }
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        artifact_dir / "manifest.json",
        {
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "source_path": str(pdf_path),
            "page_count": page_count,
            "split": True,
            "parser": "mineru_api_precise",
            "batch_id": batch_id,
            "parts": parts,
        },
    )


def _mineru_page_ranges(page_count: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + MINERU_MAX_PAGES_PER_FILE - 1, page_count))
        for start in range(1, page_count + 1, MINERU_MAX_PAGES_PER_FILE)
    ]


def _mineru_upload_url(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("url", "file_url", "upload_url"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return ""


def _wait_for_mineru_results(api_key: str, batch_id: str, expected_count: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        payload = _mineru_json("GET", f"{MINERU_API_BASE_URL}/extract-results/batch/{batch_id}", api_key)
        data = payload.get("data") if isinstance(payload, dict) else None
        results = data.get("extract_result") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise DocumentServiceError("MinerU returned invalid extraction results")
        failed = [item for item in results if isinstance(item, dict) and str(item.get("state")) == "failed"]
        if failed:
            message = str(failed[0].get("err_msg") or "MinerU parse failed")
            raise DocumentServiceError(message)
        done = [item for item in results if isinstance(item, dict) and str(item.get("state")) == "done"]
        if len(done) >= expected_count:
            return sorted(done, key=lambda item: str(item.get("data_id") or item.get("file_id") or ""))
        time.sleep(3)
    raise DocumentServiceError("MinerU parse timed out")


def _download_mineru_zip(result: dict[str, Any], target_dir: Path) -> None:
    zip_url = str(result.get("full_zip_url") or "")
    if not zip_url:
        raise DocumentServiceError("MinerU result did not include full_zip_url")
    archive = _download_bytes(zip_url)
    _extract_zip(archive, target_dir)


def _normalize_content_list(target_dir: Path) -> None:
    target = target_dir / "content_list_v2.json"
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
        _write_json(target, _normalize_content_payload(payload))
        return
    candidates = sorted(target_dir.rglob("*content_list*.json"))
    if not candidates:
        raise DocumentServiceError("MinerU output is missing content_list JSON")
    candidate = next((path for path in candidates if "content_list_v2" in path.name), candidates[0])
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    _write_json(target, _normalize_content_payload(payload))


def _normalize_content_payload(payload: Any) -> Any:
    if not isinstance(payload, list) or not payload:
        return payload
    if all(isinstance(item, list) for item in payload):
        return payload
    if not all(isinstance(item, dict) for item in payload):
        return payload

    page_indexes = [_page_index(item) for item in payload]
    pages: list[list[dict[str, Any]]] = [[] for _ in range(max(page_indexes) + 1)]
    for item, page_index in zip(payload, page_indexes, strict=True):
        pages[page_index].append(item)
    return pages


def _page_index(item: dict[str, Any]) -> int:
    try:
        return max(int(item.get("page_idx") or 0), 0)
    except (TypeError, ValueError):
        return 0


def _mineru_json(method: str, url: str, api_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise DocumentServiceError(f"MinerU request failed: {exc}") from exc
    if int(parsed.get("code", -1)) != 0:
        raise DocumentServiceError(str(parsed.get("msg") or "MinerU request failed"))
    return parsed


def _mineru_put_bytes(url: str, content: bytes) -> None:
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.netloc, timeout=120)
    try:
        connection.request("PUT", path, body=content, headers={"Content-Length": str(len(content))})
        response = connection.getresponse()
        response.read()
        if response.status >= 400:
            raise DocumentServiceError(f"MinerU upload failed: HTTP {response.status}")
    except (OSError, http.client.HTTPException) as exc:
        raise DocumentServiceError(f"MinerU upload failed: {exc}") from exc
    finally:
        connection.close()


def _download_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise DocumentServiceError(f"MinerU download failed: {exc}") from exc


def _extract_zip(archive: bytes, target_dir: Path) -> None:
    root = target_dir.resolve()
    with zipfile.ZipFile(BytesIO(archive)) as zip_file:
        for member in zip_file.infolist():
            destination = (target_dir / member.filename).resolve()
            if not _is_relative_to(destination, root):
                raise DocumentServiceError("MinerU output zip contains unsafe paths")
            zip_file.extract(member, target_dir)


def _page_to_block_list(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    return [
        {
            "type": "paragraph",
            "content": {"paragraph_content": [{"type": "text", "content": text}]},
        }
    ]


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import pymupdf

        pdf = pymupdf.open(str(pdf_path))
        try:
            return int(pdf.page_count)
        finally:
            pdf.close()
    except Exception:
        return 0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _build_page_label_map(pdf: Any) -> dict[int, str]:
    """Build a mapping from 1-based physical page number to logical page label."""
    labels: dict[int, str] = {}
    try:
        page_labels = pdf.get_page_labels()
    except Exception:
        return labels
    if not page_labels:
        return labels
    for idx, label in enumerate(page_labels):
        if isinstance(label, dict):
            text = label.get("prefix", "") + str(label.get("start", idx + 1))
            labels[idx + 1] = text
        elif isinstance(label, str):
            labels[idx + 1] = label
    return labels


def _page_count(manifest: dict[str, Any], artifact_dir: Path) -> int:
    if isinstance(manifest.get("page_count"), int):
        return int(manifest["page_count"])
    content_path = artifact_dir / "content_list_v2.json"
    if content_path.exists():
        data = _read_json(content_path)
        return len(data) if isinstance(data, list) else 0
    layout_path = artifact_dir / "layout.json"
    if layout_path.exists():
        layout = _read_json(layout_path)
        pdf_info = layout.get("pdf_info") if isinstance(layout, dict) else None
        return len(pdf_info) if isinstance(pdf_info, list) else 0
    return 0


def _read_page_from_content_list(path: Path, page: int) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise DocumentServiceError(f"Invalid MinerU content list: {path}")
    index = page - 1
    if index < 0 or index >= len(data):
        raise DocumentServiceError(f"Page {page} is outside parsed content: {path}")
    page_items = data[index]
    return page_items if isinstance(page_items, list) else []


@lru_cache(maxsize=128)
def _read_json(path: Path) -> Any:
    if not path.exists():
        raise DocumentServiceError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _normalize_block(item: dict[str, Any]) -> dict[str, Any]:
    text = _extract_text(item.get("content")).strip()
    if not text and isinstance(item.get("text"), str):
        text = item["text"].strip()
    return {
        "type": str(item.get("type") or "block"),
        "text": text,
        "bbox": item.get("bbox") if isinstance(item.get("bbox"), list) else None,
    }


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    if isinstance(value, list):
        return "".join(_extract_text(item) for item in value)
    if isinstance(value, dict):
        chunks: list[str] = []
        for key in (
            "content",
            "paragraph_content",
            "title_content",
            "page_header_content",
            "page_footer_content",
            "page_number_content",
            "item_content",
            "table_content",
            "image_caption",
            "table_caption",
            "table_footnote",
            "list_items",
        ):
            if key in value:
                text = _extract_text(value[key])
                if text:
                    chunks.append(text)
        if "html" in value and isinstance(value["html"], str):
            plain = _html_to_text(value["html"])
            if plain:
                chunks.append(plain)
        return "\n".join(chunks) if chunks else ""
    return ""


def _html_to_text(html: str) -> str:
    """Strip HTML table tags, preserving cell content with tab/newline separators."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</?t[rdh][^>]*>", "\t", text, flags=re.IGNORECASE)
    text = re.sub(r"</?tr[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?table[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    lines = [re.sub(r"^\t+|\t+$", "", line) for line in text.splitlines()]
    lines = [re.sub(r"\t{2,}", "\t", line) for line in lines]
    return "\n".join(line for line in lines if line.strip())
