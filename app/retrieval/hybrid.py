import json
import re
from dataclasses import replace
from typing import Any, Optional

from openai import OpenAI
from rank_bm25 import BM25Okapi

from app.domain import DocumentRef, Evidence


REWRITE_SYSTEM_PROMPT = (
    "你是财报检索查询改写助手。"
    "你的任务不是回答问题，而是把用户问题改写成更适合财报检索的短查询。"
    "请保留原问题中的公司名、年份、指标名和关键约束。"
    "可以补充更适合检索的术语、表格名或章节名，例如“主要会计数据”“主营业务构成”“可能面对的风险”。"
    "请只返回 JSON，格式为 {\"queries\": [\"...\", \"...\"]}。"
    "最多返回 2 条额外查询，不要返回原问题本身，不要解释。"
)

ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
CJK_SEGMENT_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def _clean_json_content(content: str) -> str:
    cleaned = content.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = [line.rstrip() for line in cleaned.splitlines()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_query_list(query: str, rewrites: list[str], *, max_queries: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in [query, *rewrites]:
        value = candidate.strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
        if len(normalized) >= max_queries:
            break
    return normalized


def tokenize_for_lexical_search(text: str) -> list[str]:
    tokens: list[str] = []
    lowered = text.lower()
    tokens.extend(ASCII_TOKEN_PATTERN.findall(lowered))
    for segment in CJK_SEGMENT_PATTERN.findall(text):
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def _matches_filters(evidence: Evidence, filters: Optional[dict[str, Any]]) -> bool:
    if not filters:
        return True
    for key, value in filters.items():
        candidate = getattr(evidence, key, None)
        if isinstance(value, dict) and "$in" in value:
            allowed = value.get("$in")
            if not isinstance(allowed, list) or candidate not in allowed:
                return False
            continue
        if candidate != value:
            return False
    return True


class LLMQueryRewriter:
    def __init__(self, client: OpenAI, model: str, max_rewrites: int = 2):
        self.client = client
        self.model = model
        self.max_rewrites = max_rewrites

    def rewrite(self, query: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message
        return self._parse_response(message.content or "")

    def _parse_response(self, content: str) -> list[str]:
        parsed = json.loads(_clean_json_content(content))
        raw_queries = parsed.get("queries", parsed.get("rewrites", []))
        if not isinstance(raw_queries, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_queries:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if not value:
                continue
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(value)
            if len(normalized) >= self.max_rewrites:
                break
        return normalized


class LexicalRetriever:
    def __init__(self, dense_retriever):
        self.dense_retriever = dense_retriever
        self._documents: list[Evidence] | None = None
        self._bm25: BM25Okapi | None = None

    def invalidate_cache(self) -> None:
        self._documents = None
        self._bm25 = None

    def _ensure_corpus(self) -> None:
        if self._documents is not None and self._bm25 is not None:
            return
        documents = self.dense_retriever.get_all_documents()
        tokenized_corpus = [
            tokenize_for_lexical_search(
                "\n".join([*document.section_path, document.text]).strip()
            )
            for document in documents
        ]
        self._documents = documents
        if not tokenized_corpus or not any(tokenized_corpus):
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(
            [tokens if tokens else ["__empty__"] for tokens in tokenized_corpus]
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:
        if top_k <= 0:
            return []

        self._ensure_corpus()
        if self._bm25 is None or self._documents is None:
            return []

        query_tokens = tokenize_for_lexical_search(query)
        if not query_tokens:
            return []

        scored_results: list[tuple[float, Evidence]] = []
        for score, document in zip(self._bm25.get_scores(query_tokens), self._documents):
            if not _matches_filters(document, filters):
                continue
            scored_results.append((float(score), replace(document, score=float(score))))

        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1].page if item[1].page is not None else float("inf"),
                item[1].chunk_id,
            )
        )
        return [document for _, document in scored_results[:top_k]]


class HybridRetriever:
    def __init__(
        self,
        dense_retriever,
        query_rewriter: LLMQueryRewriter,
        lexical_retriever: Optional[LexicalRetriever] = None,
        *,
        dense_weight: float = 2.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.query_rewriter = query_rewriter
        self.lexical_retriever = lexical_retriever or LexicalRetriever(dense_retriever)
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        self._last_retrieval_queries: list[str] = []

    def _build_queries(self, query: str) -> list[str]:
        rewrites: list[str] = []
        try:
            rewrites = self.query_rewriter.rewrite(query)
        except Exception:
            rewrites = []
        return _normalize_query_list(query, rewrites, max_queries=3)

    def search(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Evidence]:
        queries = self._build_queries(query)
        self._last_retrieval_queries = queries
        candidate_k = max(20, top_k * 5)
        fused: dict[str, dict[str, Any]] = {}

        for query_variant in queries:
            dense_results = self.dense_retriever.search(
                query_variant,
                top_k=candidate_k,
                filters=filters,
            )
            lexical_results = self.lexical_retriever.search(
                query_variant,
                top_k=candidate_k,
                filters=filters,
            )
            self._accumulate_results(fused, dense_results, weight=self.dense_weight, channel="dense")
            self._accumulate_results(fused, lexical_results, weight=self.lexical_weight, channel="lexical")

        ranked_results = sorted(
            fused.values(),
            key=lambda item: (
                -item["score"],
                item["best_dense_rank"],
                item["best_lexical_rank"],
                item["evidence"].page if item["evidence"].page is not None else float("inf"),
                item["evidence"].chunk_id,
            ),
        )
        return [
            replace(item["evidence"], score=item["score"])
            for item in ranked_results[:top_k]
        ]

    def _accumulate_results(
        self,
        fused: dict[str, dict[str, Any]],
        results: list[Evidence],
        *,
        weight: float,
        channel: str,
    ) -> None:
        rank_field = f"best_{channel}_rank"
        default_rank = float("inf")
        for rank, result in enumerate(results, start=1):
            entry = fused.setdefault(
                result.chunk_id,
                {
                    "evidence": result,
                    "score": 0.0,
                    "best_dense_rank": default_rank,
                    "best_lexical_rank": default_rank,
                },
            )
            entry["score"] += weight / (self.rrf_k + rank)
            entry[rank_field] = min(entry[rank_field], rank)

    def list_documents(self) -> list[DocumentRef]:
        return self.dense_retriever.list_documents()

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.dense_retriever.embed_chunks(chunks)

    def upsert_embedded_chunks(self, embedded_chunks: list[dict[str, Any]]) -> None:
        self.dense_retriever.upsert_embedded_chunks(embedded_chunks)
        self.invalidate_cache()

    def index_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        embedded_chunks = self.dense_retriever.index_chunks(chunks)
        self.invalidate_cache()
        return embedded_chunks

    def delete_document(self, doc_id: str) -> None:
        self.dense_retriever.delete_document(doc_id)
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        invalidate = getattr(self.lexical_retriever, "invalidate_cache", None)
        if callable(invalidate):
            invalidate()

    def get_last_retrieval_queries(self) -> list[str]:
        return list(self._last_retrieval_queries)

    def close(self) -> None:
        close = getattr(self.dense_retriever, "close", None)
        if callable(close):
            close()
