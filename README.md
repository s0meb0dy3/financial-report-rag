# Financial Report RAG

一个面向财报 PDF 的最小 RAG 学习项目。

目前已经支持：

* PDF 文本提取与切块
* embedding 生成与相似度检索
* 基于检索结果回答问题
* 返回页码引用
* 命令行提问

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
CHAT_MODEL=qwen/qwen3.5-9b
```

## 使用方式

1. 准备 PDF，并用 [`ingest.py`](/Users/peteryao/projects/CaibaoAgent/ingest.py) 生成 chunks

```bash
uv run python ingest.py
```

2. 用 [`retriever.py`](/Users/peteryao/projects/CaibaoAgent/retriever.py) 生成 embeddings

```bash
uv run python retriever.py
```

3. 用 [`agent.py`](/Users/peteryao/projects/CaibaoAgent/agent.py) 提问

```bash
uv run python agent.py "贵州茅台2024年的营业总收入是多少？"
```

也可以查看帮助：

```bash
uv run python agent.py --help
```

4. 用 [`eval.py`](/Users/peteryao/projects/CaibaoAgent/eval.py) 跑固定问题集评测

```bash
uv run python eval.py
```

评测问题集在 [`data/eval/questions.json`](/Users/peteryao/projects/CaibaoAgent/data/eval/questions.json)，结果默认写到 `data/eval/results/latest.json`。

## 测试

```bash
python3 -m unittest tests/test_agent.py tests/test_retriever.py tests/test_ingest.py tests/test_eval.py -v
```
