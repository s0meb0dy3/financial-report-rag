# Financial Report RAG

一个面向财报 PDF 的最小 RAG 学习项目，现已支持多文档索引和基于 ChromaDB 的本地持久化检索。

目前已经支持：

* 多个 PDF 的文本提取与切块
* OpenRouter embedding 生成
* ChromaDB 向量持久化
* 基于检索结果回答问题
* 返回文档名与页码引用
* 命令行提问
* 固定问题集评测

## 环境准备

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## 配置

复制 `.env.example` 为 `.env`，然后填写你的 OpenRouter key。

```env
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
CHAT_MODEL=qwen/qwen3.6-plus:free
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=financial-report-chunks
```

## 使用方式

1. 准备 PDF

推荐放到 `data/raw/` 目录下。`ingest.py` 默认会优先扫描 `data/raw/`，如果目录为空，会回退扫描项目根目录下的 PDF 文件。

2. 生成多文档 chunks

```bash
uv run python ingest.py
```

默认输出到 [`data/processed/chunks.json`](/Users/peteryao/projects/CaibaoAgent/data/processed/chunks.json)。

3. 生成 embeddings 并写入 ChromaDB

```bash
uv run python retriever.py
```

默认会把向量写入 `data/chroma/` 下的 ChromaDB 持久化目录。

4. 用 [`agent.py`](/Users/peteryao/projects/CaibaoAgent/agent.py) 提问

```bash
uv run python agent.py "贵州茅台2024年的营业总收入是多少？"
```

如果只想在某一份文档内检索，可以传 `--doc-id`。

5. 跑固定问题集评测

```bash
uv run python eval.py
```

评测问题集在 [`data/eval/questions.json`](/Users/peteryao/projects/CaibaoAgent/data/eval/questions.json)，结果默认写到 `data/eval/results/latest.json`。

## 当前流程

```text
PDFs -> chunks.json -> OpenRouter embeddings -> ChromaDB -> retrieve -> answer -> citations -> eval
```

## 测试

```bash
uv run python -m unittest tests.test_ingest tests.test_retriever tests.test_vector_store tests.test_agent tests.test_eval -v
```
