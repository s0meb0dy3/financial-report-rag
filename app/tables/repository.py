import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi


DEFAULT_TABLES_PATH = "data/processed/tables.json"
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
        return cls(path=path)

    def __init__(self, path: Path, records: Optional[list[dict[str, Any]]] = None):
        self.path = Path(path)
        self.records = records if records is not None else self._load_records()
        self._search_texts = [self._build_search_text(record) for record in self.records]
        corpus = [tokenize_table_text(text) or ["__empty__"] for text in self._search_texts]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

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
        doc_id: str,
        query: str | None = None,
        statement_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not doc_id or top_k <= 0:
            return []

        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, record in enumerate(self.records):
            if record.get("doc_id") != doc_id:
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

    def get_table(self, *, doc_id: str, table_id: str) -> dict[str, Any] | None:
        for record in self.records:
            if record.get("doc_id") != doc_id:
                continue
            if record.get("table_id") != table_id:
                continue
            return self._table_payload(record)
        return None
