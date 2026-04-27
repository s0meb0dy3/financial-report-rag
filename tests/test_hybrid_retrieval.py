import unittest
from unittest.mock import MagicMock

from app.domain import DocumentRef, Evidence
from app.retrieval import HybridRetriever, LLMQueryRewriter, LexicalRetriever


def make_evidence(
    *,
    chunk_id: str,
    doc_id: str = "moutai",
    doc_name: str = "茅台2024年年度报告完整版.pdf",
    page: int,
    text: str,
    section_path: list[str],
    score: float = 0.0,
    chunk_type: str = "paragraph",
) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        source_path=f"/tmp/{doc_name}",
        page=page,
        page_start=page,
        page_end=page,
        text=text,
        section_path=section_path,
        chunk_type=chunk_type,
        score=score,
    )


class StubDenseRetriever:
    def __init__(self, *, search_map: dict[str, list[Evidence]], documents: list[Evidence]):
        self.search_map = search_map
        self.documents = documents
        self.calls: list[tuple[str, int, dict | None]] = []
        self.closed = False

    def search(self, query: str, top_k: int = 3, filters: dict | None = None) -> list[Evidence]:
        self.calls.append((query, top_k, filters))
        results = self.search_map.get(query, [])
        if filters:
            results = [
                item
                for item in results
                if all(getattr(item, key, None) == value for key, value in filters.items())
            ]
        return results[:top_k]

    def get_all_documents(self, filters: dict | None = None) -> list[Evidence]:
        if not filters:
            return list(self.documents)
        return [
            item
            for item in self.documents
            if all(getattr(item, key, None) == value for key, value in filters.items())
        ]

    def list_documents(self) -> list[DocumentRef]:
        seen: set[tuple[str, str]] = set()
        documents: list[DocumentRef] = []
        for item in self.documents:
            key = (item.doc_id, item.doc_name)
            if key in seen:
                continue
            seen.add(key)
            documents.append(DocumentRef(doc_id=item.doc_id, doc_name=item.doc_name))
        return documents

    def close(self) -> None:
        self.closed = True


class StubRewriter:
    def __init__(self, rewrites: dict[str, list[str]] | None = None, *, error: Exception | None = None):
        self.rewrites = rewrites or {}
        self.error = error

    def rewrite(self, query: str) -> list[str]:
        if self.error is not None:
            raise self.error
        return list(self.rewrites.get(query, []))


class QueryRewriteTests(unittest.TestCase):
    def test_query_rewriter_parses_json_queries(self) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"queries": ["主要会计数据 经营活动产生的现金流量净额", "经营活动现金流量净额"]}'
                    )
                )
            ]
        )
        rewriter = LLMQueryRewriter(client, "test-model")

        rewrites = rewriter.rewrite("2024年经营活动产生的现金流量净额是多少？")

        self.assertEqual(
            rewrites,
            ["主要会计数据 经营活动产生的现金流量净额", "经营活动现金流量净额"],
        )


class LexicalRetrieverTests(unittest.TestCase):
    def test_lexical_retriever_uses_section_path_tokens(self) -> None:
        correct = make_evidence(
            chunk_id="risk-correct",
            page=22,
            section_path=["可能面对的风险"],
            text="一是宏观经济风险；二是安全风险；三是舆情风险；四是环境保护风险。",
        )
        noise = make_evidence(
            chunk_id="risk-noise",
            page=122,
            section_path=["金融工具风险"],
            text="公司的主要金融工具导致的主要风险是信用风险、流动风险、汇率风险及利率风险。",
        )
        dense_retriever = StubDenseRetriever(search_map={}, documents=[correct, noise])
        lexical = LexicalRetriever(dense_retriever)

        results = lexical.search("可能面对的风险 宏观经济风险 安全风险", top_k=2)

        self.assertEqual([item.chunk_id for item in results], ["risk-correct", "risk-noise"])


class HybridRetrieverTests(unittest.TestCase):
    def test_hybrid_retriever_falls_back_to_raw_query_when_rewriter_fails(self) -> None:
        document = make_evidence(
            chunk_id="doc-a-page-1",
            doc_id="doc-a",
            doc_name="doc-a.pdf",
            page=1,
            section_path=["第一章"],
            text="营业总收入 100 亿元。",
        )
        dense_retriever = StubDenseRetriever(
            search_map={"营业总收入是多少？": [document]},
            documents=[document],
        )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            query_rewriter=StubRewriter(error=ValueError("bad json")),
        )

        results = retriever.search("营业总收入是多少？", top_k=1)

        self.assertEqual([item.chunk_id for item in results], ["doc-a-page-1"])
        self.assertEqual(retriever.get_last_retrieval_queries(), ["营业总收入是多少？"])

    def test_hybrid_retriever_applies_filters_and_tracks_queries(self) -> None:
        doc_a = make_evidence(
            chunk_id="doc-a-page-1",
            doc_id="doc-a",
            doc_name="doc-a.pdf",
            page=1,
            section_path=["第一章"],
            text="营业总收入 100 亿元。",
        )
        doc_b = make_evidence(
            chunk_id="doc-b-page-1",
            doc_id="doc-b",
            doc_name="doc-b.pdf",
            page=1,
            section_path=["第一章"],
            text="营业总收入 200 亿元。",
        )
        dense_retriever = StubDenseRetriever(
            search_map={
                "营业总收入是多少？": [doc_a, doc_b],
                "主要会计数据 营业总收入": [doc_b, doc_a],
            },
            documents=[doc_a, doc_b],
        )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            query_rewriter=StubRewriter(
                rewrites={"营业总收入是多少？": ["主要会计数据 营业总收入"]}
            ),
        )

        results = retriever.search("营业总收入是多少？", top_k=2, filters={"doc_id": "doc-b"})

        self.assertEqual([item.doc_id for item in results], ["doc-b"])
        self.assertEqual(
            retriever.get_last_retrieval_queries(),
            ["营业总收入是多少？", "主要会计数据 营业总收入"],
        )
        self.assertEqual(
            dense_retriever.calls,
            [
                ("营业总收入是多少？", 20, {"doc_id": "doc-b"}),
                ("主要会计数据 营业总收入", 20, {"doc_id": "doc-b"}),
            ],
        )

    def test_hybrid_retriever_applies_multi_document_filter_to_lexical_results(self) -> None:
        doc_a = make_evidence(
            chunk_id="doc-a-page-1",
            doc_id="doc-a",
            page=1,
            section_path=["第一章"],
            text="营业总收入 100 亿元。",
        )
        doc_b = make_evidence(
            chunk_id="doc-b-page-1",
            doc_id="doc-b",
            page=1,
            section_path=["第一章"],
            text="营业总收入 200 亿元。",
        )
        doc_c = make_evidence(
            chunk_id="doc-c-page-1",
            doc_id="doc-c",
            page=1,
            section_path=["第一章"],
            text="营业总收入 300 亿元。",
        )
        dense_retriever = StubDenseRetriever(
            search_map={"营业总收入是多少？": [doc_a, doc_b, doc_c]},
            documents=[doc_a, doc_b, doc_c],
        )
        retriever = HybridRetriever(dense_retriever=dense_retriever, query_rewriter=StubRewriter())

        results = retriever.search(
            "营业总收入是多少？",
            top_k=3,
            filters={"doc_id": {"$in": ["doc-a", "doc-c"]}},
        )

        self.assertEqual([item.doc_id for item in results], ["doc-a", "doc-c"])

    def test_hybrid_retriever_regression_queries_rank_expected_chunks(self) -> None:
        cash_correct = make_evidence(
            chunk_id="cash-correct",
            page=5,
            chunk_type="table",
            section_path=["主要会计数据"],
            text="经营活动产生的现金流量净额 92,463,692,168.43 元。",
        )
        cash_noise = make_evidence(
            chunk_id="cash-noise",
            page=68,
            chunk_type="table",
            section_path=["母公司现金流量表"],
            text="经营活动产生的现金流量净额 36,826,593,268.70 元。",
        )
        business_correct = make_evidence(
            chunk_id="business-correct",
            page=16,
            chunk_type="table",
            section_path=["按不同类型披露公司主营业务构成"],
            text="茅台酒 营业收入 145,928,075,955.31 元。",
        )
        business_noise = make_evidence(
            chunk_id="business-noise",
            page=55,
            section_path=["关键审计事项"],
            text="主营业务收入为人民币 17,061,183.81 万元。",
        )
        risk_correct = make_evidence(
            chunk_id="risk-correct",
            page=22,
            section_path=["可能面对的风险"],
            text="一是宏观经济风险；二是安全风险；三是舆情风险；四是环境保护风险。",
        )
        risk_noise = make_evidence(
            chunk_id="risk-noise",
            page=122,
            section_path=["金融工具风险"],
            text="公司的主要风险是信用风险、流动风险、汇率风险及利率风险。",
        )

        dense_retriever = StubDenseRetriever(
            search_map={
                "2024年经营活动产生的现金流量净额是多少？": [cash_noise, cash_correct],
                "主要会计数据 经营活动产生的现金流量净额": [cash_correct, cash_noise],
                "2024年茅台酒的主营业务收入是多少？": [business_noise, business_correct],
                "按不同类型披露公司主营业务构成 茅台酒 营业收入": [
                    business_correct,
                    business_noise,
                ],
                "公司在年报中提到的主要风险有哪些？": [risk_noise, risk_correct],
                "可能面对的风险 宏观经济风险 安全风险 舆情风险 环境保护风险": [
                    risk_correct,
                    risk_noise,
                ],
            },
            documents=[
                cash_correct,
                cash_noise,
                business_correct,
                business_noise,
                risk_correct,
                risk_noise,
            ],
        )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            query_rewriter=StubRewriter(
                rewrites={
                    "2024年经营活动产生的现金流量净额是多少？": [
                        "主要会计数据 经营活动产生的现金流量净额"
                    ],
                    "2024年茅台酒的主营业务收入是多少？": [
                        "按不同类型披露公司主营业务构成 茅台酒 营业收入"
                    ],
                    "公司在年报中提到的主要风险有哪些？": [
                        "可能面对的风险 宏观经济风险 安全风险 舆情风险 环境保护风险"
                    ],
                }
            ),
        )

        cases = [
            ("2024年经营活动产生的现金流量净额是多少？", "cash-correct"),
            ("2024年茅台酒的主营业务收入是多少？", "business-correct"),
            ("公司在年报中提到的主要风险有哪些？", "risk-correct"),
        ]

        for query, expected_chunk_id in cases:
            with self.subTest(query=query):
                results = retriever.search(query, top_k=3, filters={"doc_id": "moutai"})
                self.assertEqual(results[0].chunk_id, expected_chunk_id)

    def test_hybrid_retriever_close_delegates_to_dense_retriever(self) -> None:
        document = make_evidence(
            chunk_id="doc-a-page-1",
            doc_id="doc-a",
            doc_name="doc-a.pdf",
            page=1,
            section_path=["第一章"],
            text="营业总收入 100 亿元。",
        )
        dense_retriever = StubDenseRetriever(
            search_map={"营业总收入是多少？": [document]},
            documents=[document],
        )
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            query_rewriter=StubRewriter(),
        )

        retriever.close()

        self.assertTrue(dense_retriever.closed)


if __name__ == "__main__":
    unittest.main()
