# Fintell

极简前后端分离 Chatbox。后端负责 OpenAI-compatible 聊天、流式输出和 SQLite 会话历史；前端负责单页对话体验。RAG、检索和索引代码保留为后端可复用模块，但当前聊天链路默认不把任何工具接给模型。

## 功能

- Chat：调用 OpenAI-compatible chat model，支持流式 `reasoning_content`。
- History：SQLite 持久化默认会话，刷新页面后可恢复历史。
- Frontend：金融风格单页 chatbox，支持 Markdown 和 GFM 表格渲染。
- Usage：展示 token/context 占用，前端用 10 格占用条表达上下文压力。
- RAG modules：保留本地 PDF ingest、向量 index、混合检索和 RAG service，便于后续重新接入工具。

## 环境变量

复制 `.env.example` 后填入密钥：

```bash
cp .env.example .env
```

最小聊天配置：

```bash
MIMO_API_KEY=your-mimo-api-key
CHAT_BASE_URL=https://api.xiaomimimo.com/v1
CHAT_MODEL=mimo-v2.5-pro
CHAT_THINKING_ENABLED=true
CHAT_PASS_REASONING_HISTORY=true
CHAT_STREAM_INCLUDE_USAGE=false
CONTEXT_WINDOW_TOKENS=128000
SESSION_DB_PATH=data/sessions.sqlite3
TAVILY_API_KEY=your-tavily-api-key
```

`CHAT_*` 优先于旧的 `OPENROUTER_*` 聊天配置；只设置 `MIMO_API_KEY` 且未设置 `CHAT_BASE_URL` 时，后端默认使用 `https://api.xiaomimimo.com/v1`。MiMo thinking 模式下，历史 assistant 的 `reasoning_content` 会随 SQLite 历史一起回传给后续请求。设置 `TAVILY_API_KEY` 后，模型可以自行决定是否调用 `tavily_search` 做网页搜索。

如需继续准备本地 RAG 索引，再配置 embedding 和 MinerU：

```bash
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
CHROMA_PERSIST_DIR=data/chroma
CHROMA_COLLECTION_NAME=financial-report-chunks
MINERU_API_TOKEN=your-mineru-api-token
MINERU_BASE_URL=https://mineru.net
MINERU_MODEL_VERSION=vlm
MINERU_LANGUAGE=ch
```

## 准备数据

聊天不依赖索引；以下命令只在需要本地 RAG 数据时运行。

```bash
uv sync
uv run python main.py ingest --input-dir data/raw --artifact-dir data/processed/mineru
uv run python main.py index --chunks-path data/processed/chunks.json
```

本地数据默认落在 `data/processed/`、`data/chroma/` 和 `data/sessions.sqlite3`。

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

打开 `http://127.0.0.1:5173/`。前端会在浏览器 `localStorage` 中保存一个默认 `session_id`。

## API

- `GET /health`：健康检查。
- `GET /sessions`：列出已有会话。
- `GET /sessions/{session_id}`：恢复单个会话历史。
- `POST /chat`：非流式问答，返回 `session_id`、`answer`、`citations`、`tool_results`、`reasoning_content`、`usage`。
- `POST /chat/stream`：SSE 问答，事件包括 `session`、`status`、`reasoning_delta`、`tool_call`、`tool_result`、`answer_delta`、`usage`、`final`、`error`。

`POST /chat` 示例：

```json
{
  "question": "NVIDIA T4 GPU 是什么？",
  "session_id": "local-default"
}
```

`usage.context_ratio` 会驱动前端 10 格上下文占用条；低占用为绿色，中等为黄色，高占用为红色。

## 代码结构

- `app/api.py`：FastAPI HTTP/SSE API。
- `app/chat_service.py`：轻量聊天流程，负责历史拼接、模型调用、usage 解析和 turn 持久化。
- `app/factory.py`：从环境变量构建 `ChatService` 的小型工厂。
- `app/tools/`：OpenAI-compatible tool 抽象，目前包含 Tavily web search。
- `app/session/`：SQLite session 和 turn 历史。
- `app/ingestion/`：可选 PDF 解析和 chunk 生成。
- `app/retrieval/`：可选向量检索、BM25、query rewrite 和混合融合。
- `app/rag/`：可选 RAG evidence/citation 封装。
- `app/tables/`：可选表格索引读取。
- `frontend/`：React 单页 chatbox。

新增工具时，优先在 `app/tools/` 下实现一个类，提供：

- `name`：暴露给模型的唯一工具名。
- `aliases`：可选，仅用于兼容模型输出的旧名称，不会重复暴露给模型。
- `schema()`：OpenAI-compatible tool schema。
- `run(arguments)`：执行工具并返回结构化结果，可选包含 `citations`。

然后在 `app/factory.py` 注册到 `ChatService(tools=[...])`。工具 schema、别名、MiMo 文本工具调用兼容和执行错误都会由 `ToolRegistry` 统一处理。

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
