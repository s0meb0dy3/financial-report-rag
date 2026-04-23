# CaibaoAgent 项目架构

## 1. 当前主流程

项目现在统一走 MinerU 解析链路：

```text
PDF
-> MinerU precision 解析
-> content_list_v2.json / full.md / result.zip
-> ParsedDocument
-> StructuredMineruChunker
-> chunks.json
-> OpenRouter Embeddings
-> ChromaDB
-> search_reports
-> Agent Loop
-> Answer / Eval
```

这条链路的目标是尽量保留财报里的章节结构、页码信息、表格内容和可追溯来源，让检索和引用都更稳定。

## 2. CLI 入口

- [`main.py`](/Users/peteryao/projects/CaibaoAgent/main.py)

统一提供四个子命令：

- `ingest`
- `index`
- `chat`
- `eval`

## 3. Ingestion

- [`app/ingestion/service.py`](/Users/peteryao/projects/CaibaoAgent/app/ingestion/service.py)
- [`app/ingestion/parsers.py`](/Users/peteryao/projects/CaibaoAgent/app/ingestion/parsers.py)
- [`app/ingestion/chunking.py`](/Users/peteryao/projects/CaibaoAgent/app/ingestion/chunking.py)
- [`app/ingestion/types.py`](/Users/peteryao/projects/CaibaoAgent/app/ingestion/types.py)

当前 parser 是 `MineruPdfParser`，会：

- 通过 MinerU API 解析 PDF
- 复用 `data/processed/mineru/<doc_id>/` 下的缓存产物
- 优先读取 `content_list_v2.json`
- 兼容旧版 `*_content_list.json`
- 产出标准化 `ParsedDocument`

当前 chunker 是 `StructuredMineruChunker`，会：

- 用标题维护 `section_path`
- 聚合同一章节下的正文段落
- 对超长正文按换行和标点做受控切分
- 将表格保留为独立 chunk
- 尝试合并跨页续表

默认 ingest 产物：

- `data/processed/chunks.json`
- `data/processed/mineru/<doc_id>/result.zip`
- `data/processed/mineru/<doc_id>/content_list_v2.json`
- `data/processed/mineru/<doc_id>/full.md`

## 4. Retrieval

- [`app/retrieval/retriever.py`](/Users/peteryao/projects/CaibaoAgent/app/retrieval/retriever.py)
- [`app/retrieval/vector_store.py`](/Users/peteryao/projects/CaibaoAgent/app/retrieval/vector_store.py)

检索层会：

- 对 `chunks.json` 做 batch embedding
- 优先使用 `embedding_text`
- 将向量写入本地 `data/chroma/`
- 通过 `search_reports` 返回带页码和文档名的证据

## 5. Agent Runtime

- [`app/agent.py`](/Users/peteryao/projects/CaibaoAgent/app/agent.py)
- [`app/runtime/openai_client.py`](/Users/peteryao/projects/CaibaoAgent/app/runtime/openai_client.py)
- [`app/tools/financial_reports.py`](/Users/peteryao/projects/CaibaoAgent/app/tools/financial_reports.py)

对话时，`AgentLoop` 会驱动模型调用 `search_reports`，把检索到的证据反馈回模型，再生成最终回答和引用。

## 6. Eval

- [`app/eval.py`](/Users/peteryao/projects/CaibaoAgent/app/eval.py)
- [`data/eval/questions.json`](/Users/peteryao/projects/CaibaoAgent/data/eval/questions.json)

评测会读取固定问题集，调用当前 agent 产出答案，再让同一模型按标准答案和引用页码做 JSON 判分。

## 7. 现阶段约束

- 现在的索引目录 `data/chroma/` 是本地持久化数据，需要在切换解析策略时主动清空或重建。
- `ingest` 和 `index` 仍然是两个独立步骤；重新解析 PDF 后，需要重新执行 `index`。
- 当前主要面向 CLI 和 Python 调用，还没有封装成 HTTP API 服务。
