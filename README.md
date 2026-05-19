# Fintell

极简财报 RAG Chat：先用 CLI 准备本地财报索引，再启动 FastAPI + React 单页聊天框。后端固定执行 `retrieve -> answer -> persist`，不再保留多工具 agent loop、图表生成、上传工作台或前端文档管理。

## 功能

- CLI：解析财报 PDF、切 chunk、写入本地向量索引。
- RAG：混合检索文本 chunk，并补充表格候选摘要。
- Chat：每轮先检索证据，再调用一次 OpenAI-compatible chat model 合成答案，支持流式 `reasoning_content`。
- History：SQLite 持久化当前 chat box 的问答历史。
- Frontend：DeepSeek-like 双栏 chatbox，按思考、工具、回答分区展示，带 token/context 10 格占用条和引用页码。

## 环境变量

复制 `.env.example` 后填入密钥：

```bash
cp .env.example .env
```

核心配置。聊天模型可以直接使用 Xiaomi MiMo；embedding 可继续用 OpenRouter 或其他 OpenAI-compatible embedding provider：

```bash
MIMO_API_KEY=your-mimo-api-key
CHAT_BASE_URL=https://api.xiaomimimo.com/v1
CHAT_MODEL=mimo-v2.5-pro
CHAT_THINKING_ENABLED=true
CHAT_PASS_REASONING_HISTORY=true
CHAT_STREAM_INCLUDE_USAGE=false

OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
CONTEXT_WINDOW_TOKENS=128000
SESSION_DB_PATH=data/sessions.sqlite3
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=financial-report-chunks
```

`CHAT_*` 优先于旧的 `OPENROUTER_*` 聊天配置；只设置 `MIMO_API_KEY` 且未设置 `CHAT_BASE_URL` 时，后端会默认使用 `https://api.xiaomimimo.com/v1`。小米 MiMo thinking 模式下，历史 assistant 的 `reasoning_content` 会随 SQLite 历史一起回传给后续请求。MiMo 文档未声明 `stream_options.include_usage`，所以示例里关闭该参数，前端上下文条会使用后端估算 token。

如果要通过 MinerU 解析 PDF，再配置 `MINERU_API_TOKEN`、`MINERU_BASE_URL`、`MINERU_MODEL_VERSION` 和 `MINERU_LANGUAGE`。

## 准备数据

安装 Python 依赖：

```bash
uv sync
```

解析 PDF：

```bash
uv run python main.py ingest --input-dir data/raw --artifact-dir data/processed/mineru
```

建立向量索引：

```bash
uv run python main.py index --chunks-path data/processed/chunks.json
```

文档管理不在前端暴露；本地数据默认落在 `data/processed/`、`data/chroma/` 和 `data/sessions.sqlite3`。

## 启动

后端：

```bash
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173/`。前端会在浏览器 `localStorage` 中保存一个默认 `session_id`，刷新后通过 SQLite 恢复该会话历史。

## API

- `GET /health`：健康检查。
- `POST /rag/retrieve`：RAG 工具接口，返回文本证据、表格候选、引用和检索元数据。
- `POST /chat`：非流式问答，返回 `session_id`、`answer`、`citations`、`reasoning_content`、`usage`。
- `POST /chat/stream`：SSE 问答，事件包括 `session`、`tool`、`status`、`reasoning_delta`、`answer_delta`、`usage`、`final`、`error`。
- `GET /sessions/{session_id}`：恢复单个会话历史，包括 assistant 的 `reasoning_content`、RAG 工具状态和 usage。

`POST /chat` 示例：

```json
{
  "question": "贵州茅台 2024 年营业总收入是多少？",
  "session_id": "local-default",
  "top_k": 5
}
```

`usage` 中的 `context_ratio` 会驱动前端 10 格像素风上下文占用条；低占用为绿色，中等为黄色，高占用为红色。

`POST /rag/retrieve` 示例：

```json
{
  "query": "贵州茅台 2024 年营业总收入",
  "top_k": 5,
  "include_tables": true
}
```

## 代码结构

- `app/api.py`：极简 HTTP/SSE API。
- `app/chat_service.py`：固定的 RAG-first 问答流程。
- `app/rag/`：RAG 工具服务和证据结构。
- `app/retrieval/`：向量检索、BM25、query rewrite 和混合融合。
- `app/ingestion/`：PDF 解析、chunk 生成和索引准备。
- `app/session/`：SQLite 会话和 turn 历史。
- `frontend/`：单页 React chat box。

## 验证

后端测试：

```bash
uv run python -m unittest discover -s tests
```

前端构建：

```bash
cd frontend
npm run build
```
