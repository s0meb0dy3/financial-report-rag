import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi


DEFAULT_TABLES_PATH = "data/processed/tables.json"
DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"
ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
CJK_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
STATEMENT_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "income_statement": ("利润表", "营业总收入", "净利润"),
    "balance_sheet": ("资产负债表", "资产总计", "负债合计"),
    "cash_flow": ("现金流量表", "经营活动产生的现金流量净额"),
    "key_metrics": ("主要会计数据", "主要财务指标"),
}


def tokenize_table_text(text: str) -> list[str]:
    tokens: list[str] = []
    lowered = text.lower()
    tokens.extend(ASCII_TOKEN_PATTERN.findall(lowered))
    for segment in CJK_SEGMENT_PATTERN.findall(text):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def _flatten_matrix(matrix: list[list[str]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in matrix if row)


def _compact_text(text: str) -> str:
    return "".join(str(text).split()).casefold()


def _page_span(record: dict[str, Any]) -> int:
    page_start = record.get("page_start")
    page_end = record.get("page_end")
    if isinstance(page_start, int) and isinstance(page_end, int):
        return max(1, page_end - page_start + 1)
    return 1


def _row_count(record: dict[str, Any]) -> int:
    value = record.get("row_count")
    if isinstance(value, int):
        return value
    matrix = record.get("matrix", [])
    return len(matrix) if isinstance(matrix, list) else 0


def _column_count(record: dict[str, Any]) -> int:
    value = record.get("column_count")
    if isinstance(value, int):
        return value
    matrix = record.get("matrix", [])
    if not isinstance(matrix, list):
        return 0
    return max((len(row) for row in matrix if isinstance(row, list)), default=0)


def _clean_cell(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _parse_pipe_matrix(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(text).splitlines():
        if "|" not in line:
            continue
        cells = [_clean_cell(cell) for cell in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


def _guess_statement_type(record: dict[str, Any]) -> str | None:
    section_text = " ".join(str(item) for item in record.get("section_path", []))
    compact = _compact_text(
        "\n".join(
            [
                str(record.get("title", "")),
                section_text,
                str(record.get("text", "")),
            ]
        )
    )
    if "主要会计数据" in compact or "主要财务指标" in compact:
        return "key_metrics"
    if "现金流量表" in compact or "现金流量净额" in compact:
        return "cash_flow"
    if "资产负债表" in compact or "资产总计" in compact or "负债合计" in compact:
        return "balance_sheet"
    if "利润表" in compact or "营业总收入" in compact or "营业收入" in compact or "净利润" in compact:
        return "income_statement"
    return None


def _record_from_table_chunk(chunk: dict[str, Any]) -> dict[str, Any] | None:
    if chunk.get("chunk_type") != "table":
        return None
    table_id = str(chunk.get("table_id") or chunk.get("chunk_id") or "").strip()
    if not table_id:
        return None
    section_path = [str(item) for item in chunk.get("section_path", []) if str(item).strip()]
    text = str(chunk.get("text", "")).strip()
    matrix = _parse_pipe_matrix(text)
    page_start = chunk.get("page_start", chunk.get("page"))
    page_end = chunk.get("page_end", chunk.get("page"))
    title = section_path[-1] if section_path else table_id
    record = {
        "table_id": table_id,
        "doc_id": chunk.get("doc_id", ""),
        "doc_name": chunk.get("doc_name", ""),
        "title": title,
        "statement_type_guess": None,
        "section_path": section_path,
        "page_start": page_start,
        "page_end": page_end,
        "preview_matrix": matrix[:8],
        "matrix": matrix,
        "footnotes_text": "",
        "text": text,
        "fragments": [
            {
                "source_chunk_id": chunk.get("chunk_id", ""),
                "page_start": page_start,
                "page_end": page_end,
                "row_count": len(matrix),
            }
        ],
        "row_count": len(matrix),
        "column_count": max((len(row) for row in matrix), default=0),
    }
    record["statement_type_guess"] = _guess_statement_type(record)
    return record


def _normalize_doc_ids(
    doc_id: str | None = None,
    doc_ids: list[str] | None = None,
) -> list[str]:
    candidates = doc_ids if doc_ids is not None else ([doc_id] if doc_id else [])
    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str):
            continue
        resolved = value.strip()
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


class JsonTableRepository:
    @classmethod
    def from_env(cls) -> "JsonTableRepository":
        project_root = Path(__file__).resolve().parents[2]
        path = Path(
            os.environ.get(
                "TABLES_PATH",
                str(project_root / DEFAULT_TABLES_PATH),
            )
        )
        chunks_path = Path(
            os.environ.get(
                "CHUNKS_PATH",
                str(project_root / DEFAULT_CHUNKS_PATH),
            )
        )
        return cls(path=path, chunks_path=chunks_path)

    def __init__(
        self,
        path: Path,
        records: Optional[list[dict[str, Any]]] = None,
        chunks_path: Path | None = None,
    ):
        self.path = Path(path)
        self.chunks_path = Path(chunks_path) if chunks_path is not None else None
        self.records = records if records is not None else self._load_records()
        self._search_texts = [self._build_search_text(record) for record in self.records]
        corpus = [tokenize_table_text(text) or ["__empty__"] for text in self._search_texts]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _load_records(self) -> list[dict[str, Any]]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        if self.chunks_path is None or not self.chunks_path.exists():
            return []
        chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if not isinstance(chunks, list):
            return []
        return [
            record
            for chunk in chunks
            if isinstance(chunk, dict)
            for record in [_record_from_table_chunk(chunk)]
            if record is not None
        ]

    @staticmethod
    def _build_search_text(record: dict[str, Any]) -> str:
        parts = [
            record.get("title", ""),
            " ".join(record.get("section_path", [])),
            record.get("statement_type_guess", "") or "",
            record.get("text", ""),
            _flatten_matrix(record.get("matrix", [])),
        ]
        return "\n".join(str(part).strip() for part in parts if str(part).strip())

    @staticmethod
    def _result_payload(record: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "table_id": record.get("table_id", ""),
            "doc_id": record.get("doc_id", ""),
            "doc_name": record.get("doc_name", ""),
            "title": record.get("title", ""),
            "statement_type_guess": record.get("statement_type_guess"),
            "section_path": list(record.get("section_path", [])),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "preview_matrix": record.get("preview_matrix", []),
            "score": round(float(score), 4),
        }

    @staticmethod
    def _table_payload(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "table_id": record.get("table_id", ""),
            "doc_id": record.get("doc_id", ""),
            "doc_name": record.get("doc_name", ""),
            "title": record.get("title", ""),
            "statement_type_guess": record.get("statement_type_guess"),
            "section_path": list(record.get("section_path", [])),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "matrix": record.get("matrix", []),
            "footnotes_text": record.get("footnotes_text", ""),
            "fragments": record.get("fragments", []),
            "row_count": _row_count(record),
            "column_count": _column_count(record),
        }

    def search_tables(
        self,
        *,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        query: str | None = None,
        statement_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        selected_doc_ids = set(_normalize_doc_ids(doc_id=doc_id, doc_ids=doc_ids))

        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, record in enumerate(self.records):
            if selected_doc_ids and record.get("doc_id") not in selected_doc_ids:
                continue
            if statement_type and record.get("statement_type_guess") != statement_type:
                continue
            candidates.append((index, record))

        if not candidates:
            return []

        query_text = (query or "").strip()
        if not query_text and statement_type:
            query_text = " ".join(STATEMENT_TYPE_HINTS.get(statement_type, (statement_type,)))
        query_tokens = tokenize_table_text(query_text)
        bm25_scores = self._bm25.get_scores(query_tokens) if self._bm25 and query_tokens else []
        compact_query = _compact_text(query_text)

        ranked: list[tuple[float, dict[str, Any]]] = []
        for index, record in candidates:
            score = float(bm25_scores[index]) if len(bm25_scores) > index else 0.0
            search_text = self._search_texts[index]
            compact_search_text = _compact_text(search_text)
            if compact_query and compact_query in compact_search_text:
                score += 3.0
            if statement_type and record.get("statement_type_guess") == statement_type:
                score += 1.0
            ranked.append((score, record))

        ranked.sort(
            key=lambda item: (
                -item[0],
                -_page_span(item[1]),
                -_row_count(item[1]),
                item[1].get("page_start") if isinstance(item[1].get("page_start"), int) else float("inf"),
                item[1].get("table_id", ""),
            )
        )
        return [
            self._result_payload(record, score)
            for score, record in ranked[:top_k]
        ]

    def get_table(self, *, table_id: str, doc_id: str | None = None) -> dict[str, Any] | None:
        for record in self.records:
            if doc_id and record.get("doc_id") != doc_id:
                continue
            if record.get("table_id") != table_id:
                continue
            return self._table_payload(record)
        return None
