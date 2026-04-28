import json
import tempfile
import unittest
from pathlib import Path

from app.tables import JsonTableRepository


class JsonTableRepositoryTests(unittest.TestCase):
    def test_falls_back_to_table_chunks_when_tables_json_is_missing(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a-chunk-1",
                "chunk_type": "table",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "section_path": ["第二节", "主要会计数据"],
                "page": 5,
                "page_start": 5,
                "page_end": 5,
                "text": "主要会计数据 | 2024年 | 2023年\n营业收入 | 100 | 80\n净利润 | 30 | 20",
            },
            {
                "chunk_id": "doc-a-chunk-2",
                "chunk_type": "paragraph",
                "doc_id": "doc-a",
                "doc_name": "doc-a.pdf",
                "text": "正文",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            tables_path = Path(temp_dir) / "tables.json"
            chunks_path = Path(temp_dir) / "chunks.json"
            chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            repository = JsonTableRepository(tables_path, chunks_path=chunks_path)

            results = repository.search_tables(doc_id="doc-a", query="营业收入")
            table = repository.get_table(table_id="doc-a-chunk-1")

        self.assertEqual(len(repository.records), 1)
        self.assertEqual(results[0]["table_id"], "doc-a-chunk-1")
        self.assertEqual(results[0]["statement_type_guess"], "key_metrics")
        self.assertEqual(table["matrix"][1], ["营业收入", "100", "80"])

    def test_search_tables_supports_statement_type_filter(self) -> None:
        records = [
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-1",
                "title": "合并现金流量表",
                "statement_type_guess": "cash_flow",
                "section_path": ["财务报告"],
                "page_start": 10,
                "page_end": 11,
                "preview_matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                "matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                "footnotes_text": "",
                "text": "| 项目 | 本期 |\n| --- | --- |\n| 经营活动产生的现金流量净额 | 100 |",
                "fragments": [{"source_element_id": "table-1", "page_start": 10, "page_end": 11, "row_count": 2}],
                "row_count": 2,
                "column_count": 2,
            },
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-2",
                "title": "合并利润表",
                "statement_type_guess": "income_statement",
                "section_path": ["财务报告"],
                "page_start": 12,
                "page_end": 13,
                "preview_matrix": [["项目", "本期"], ["营业总收入", "200"]],
                "matrix": [["项目", "本期"], ["营业总收入", "200"]],
                "footnotes_text": "",
                "text": "| 项目 | 本期 |\n| --- | --- |\n| 营业总收入 | 200 |",
                "fragments": [{"source_element_id": "table-2", "page_start": 12, "page_end": 13, "row_count": 2}],
                "row_count": 2,
                "column_count": 2,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            repository = JsonTableRepository(path)

            results = repository.search_tables(doc_id="moutai", statement_type="cash_flow")

        self.assertEqual([item["table_id"] for item in results], ["moutai-logical-table-1"])

    def test_search_tables_ranks_metric_match_above_other_tables(self) -> None:
        records = [
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-1",
                "title": "合并现金流量表",
                "statement_type_guess": "cash_flow",
                "section_path": ["财务报告"],
                "page_start": 10,
                "page_end": 11,
                "preview_matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                "matrix": [["项目", "本期"], ["经营活动产生的现金流量净额", "100"]],
                "footnotes_text": "",
                "text": "| 项目 | 本期 |\n| --- | --- |\n| 经营活动产生的现金流量净额 | 100 |",
                "fragments": [{"source_element_id": "table-1", "page_start": 10, "page_end": 11, "row_count": 2}],
                "row_count": 2,
                "column_count": 2,
            },
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-2",
                "title": "主要会计数据",
                "statement_type_guess": "key_metrics",
                "section_path": ["主要会计数据"],
                "page_start": 4,
                "page_end": 4,
                "preview_matrix": [["指标", "本期"], ["营业总收入", "200"]],
                "matrix": [["指标", "本期"], ["营业总收入", "200"]],
                "footnotes_text": "",
                "text": "| 指标 | 本期 |\n| --- | --- |\n| 营业总收入 | 200 |",
                "fragments": [{"source_element_id": "table-2", "page_start": 4, "page_end": 4, "row_count": 2}],
                "row_count": 2,
                "column_count": 2,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            repository = JsonTableRepository(path)

            results = repository.search_tables(
                doc_id="moutai",
                query="经营活动产生的现金流量净额",
                top_k=2,
            )

        self.assertEqual(results[0]["table_id"], "moutai-logical-table-1")

    def test_search_tables_supports_multiple_document_filters(self) -> None:
        records = [
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-table-1",
                "title": "主要会计数据",
                "statement_type_guess": "key_metrics",
                "section_path": ["主要会计数据"],
                "page_start": 4,
                "page_end": 4,
                "preview_matrix": [["指标", "本期"], ["营业收入", "200"]],
                "matrix": [["指标", "本期"], ["营业收入", "200"]],
                "footnotes_text": "",
                "text": "营业收入 | 200",
                "fragments": [],
                "row_count": 2,
                "column_count": 2,
            },
            {
                "doc_id": "midea",
                "doc_name": "美的集团2024年年度报告.pdf",
                "table_id": "midea-table-1",
                "title": "主要会计数据",
                "statement_type_guess": "key_metrics",
                "section_path": ["主要会计数据"],
                "page_start": 8,
                "page_end": 8,
                "preview_matrix": [["指标", "本期"], ["营业收入", "300"]],
                "matrix": [["指标", "本期"], ["营业收入", "300"]],
                "footnotes_text": "",
                "text": "营业收入 | 300",
                "fragments": [],
                "row_count": 2,
                "column_count": 2,
            },
        ]

        repository = JsonTableRepository(Path("missing.json"), records=records)

        results = repository.search_tables(doc_ids=["midea"], query="营业收入")

        self.assertEqual([item["table_id"] for item in results], ["midea-table-1"])

    def test_search_tables_prefers_wider_page_span_on_score_tie(self) -> None:
        records = [
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-1",
                "title": "母公司资产负债表",
                "statement_type_guess": "balance_sheet",
                "section_path": ["母公司资产负债表"],
                "page_start": 61,
                "page_end": 63,
                "preview_matrix": [["项目", "附注", "2024年12月31日", "2023年12月31日"]],
                "matrix": [["项目", "附注", "2024年12月31日", "2023年12月31日"]] * 10,
                "footnotes_text": "",
                "text": "| 项目 | 附注 | 2024年12月31日 | 2023年12月31日 |",
                "fragments": [{"source_element_id": "table-1", "page_start": 61, "page_end": 63, "row_count": 10}],
                "row_count": 10,
                "column_count": 4,
            },
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-2",
                "title": "母公司资产负债表",
                "statement_type_guess": "balance_sheet",
                "section_path": ["母公司资产负债表"],
                "page_start": 61,
                "page_end": 61,
                "preview_matrix": [["项目", "附注", "2024年12月31日", "2023年12月31日"]],
                "matrix": [["项目", "附注", "2024年12月31日", "2023年12月31日"]] * 3,
                "footnotes_text": "",
                "text": "| 项目 | 附注 | 2024年12月31日 | 2023年12月31日 |",
                "fragments": [{"source_element_id": "table-2", "page_start": 61, "page_end": 61, "row_count": 3}],
                "row_count": 3,
                "column_count": 4,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            repository = JsonTableRepository(path)

            results = repository.search_tables(doc_id="moutai", statement_type="balance_sheet")

        self.assertEqual(results[0]["table_id"], "moutai-logical-table-1")

    def test_get_table_returns_full_table_payload(self) -> None:
        records = [
            {
                "doc_id": "moutai",
                "doc_name": "茅台2024年年度报告完整版.pdf",
                "table_id": "moutai-logical-table-1",
                "title": "主要会计数据",
                "statement_type_guess": "key_metrics",
                "section_path": ["主要会计数据"],
                "page_start": 4,
                "page_end": 4,
                "preview_matrix": [["指标", "本期"]],
                "matrix": [["指标", "本期"], ["营业总收入", "200"]],
                "footnotes_text": "单位：元",
                "text": "| 指标 | 本期 |\n| --- | --- |\n| 营业总收入 | 200 |",
                "fragments": [{"source_element_id": "table-1", "page_start": 4, "page_end": 4, "row_count": 2}],
                "row_count": 2,
                "column_count": 2,
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tables.json"
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            repository = JsonTableRepository(path)

            table = repository.get_table(doc_id="moutai", table_id="moutai-logical-table-1")

        self.assertIsNotNone(table)
        self.assertEqual(table["matrix"][1][0], "营业总收入")
        self.assertEqual(table["footnotes_text"], "单位：元")
        self.assertEqual(table["row_count"], 2)
        self.assertEqual(table["column_count"], 2)


if __name__ == "__main__":
    unittest.main()
