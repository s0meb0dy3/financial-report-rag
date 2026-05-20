import json
import re
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Any


DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_MINERU_DIR = Path("data/processed/mineru")


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
    ):
        self.raw_dir = Path(raw_dir)
        self.mineru_dir = Path(mineru_dir)

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

    def read_page(self, doc_id: str, page: int) -> DocumentPage:
        doc = self.get_document(doc_id)
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


def _normalize_block(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": str(item.get("type") or "block"),
        "text": _extract_text(item.get("content")).strip(),
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
