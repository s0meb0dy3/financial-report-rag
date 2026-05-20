# Fintell

极简前后端分离 Chatbox。后端负责 OpenAI-compatible 聊天、流式输出、工具调用和 SQLite 会话历史；前端负责单页对话体验。当前版本不包含 RAG、PDF 解析、向量索引或文档管理，原始财报 PDF 仍保留在 `data/raw/` 供以后扩展。

## 功能

- Chat：调用 OpenAI-compatible chat model，支持流式最终回答。
- Reasoning：兼容 MiMo 的 `reasoning_content` 流式展示和历史回传。
- Tools：保留一个轻量工具抽象层，目前可选接入 Tavily web search。
- History：SQLite 持久化会话，刷新页面后可恢复历史。
- Usage：展示 token/context 占用，前端用 10 格占用条表达上下文压力。
- Frontend：金融风格单页 chatbox，支持 Markdown 和 GFM 表格渲染。

## 环境变量

复制 `.env.example` 后填入密钥：

```bash
cp .env.example .env
```

最小配置：

```bash
MIMO_API_KEY=your-mimo-api-key
CHAT_BASE_URL=https://api.xiaomimimo.com/v1
CHAT_MODEL=mimo-v2.5-pro
CHAT_THINKING_ENABLED=true
CHAT_PASS_REASONING_HISTORY=true
CHAT_STREAM_INCLUDE_USAGE=false
CONTEXT_WINDOW_TOKENS=128000
SESSION_DB_PATH=data/sessions.sqlite3
```

可选工具：

```bash
TAVILY_API_KEY=your-tavily-api-key
```

`CHAT_*` 优先于旧的 `OPENROUTER_*`/`OPENAI_*` 配置；只设置 `MIMO_API_KEY` 且未设置 `CHAT_BASE_URL` 时，后端默认使用 `https://api.xiaomimimo.com/v1`。MiMo thinking 模式下，历史 assistant 的 `reasoning_content` 会随 SQLite 历史一起回传给后续请求。

## 启动

后端：

```bash
uv sync
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

也可以用轻量 CLI：

```bash
uv run python main.py serve --reload
uv run python main.py chat "你好"
```

前端：

```bash
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173/`。

## API

- `GET /health`：健康检查。
- `GET /sessions`：列出已有会话。
- `GET /sessions/{session_id}`：恢复单个会话历史。
- `POST /chat`：非流式问答。
- `POST /chat/stream`：SSE 问答，事件包括 `session`、`status`、`reasoning_delta`、`tool_call`、`tool_result`、`answer_delta`、`usage`、`final`、`error`。

`POST /chat` 示例：

```json
{
  "question": "NVIDIA T4 GPU 是什么？",
  "session_id": "local-default"
}
```

## 代码结构

- `app/api.py`：FastAPI HTTP/SSE API。
- `app/chat_service.py`：聊天流程，负责历史拼接、模型调用、工具循环、usage 解析和 turn 持久化。
- `app/config.py`：环境变量读取。
- `app/factory.py`：从环境变量构建 `ChatService`。
- `app/session/`：SQLite session 和 turn 历史。
- `app/tools/`：OpenAI-compatible tool 抽象，目前包含 Tavily web search。
- `frontend/`：React 单页 chatbox。
- `data/raw/`：保留原始财报 PDF，当前运行链路不会读取。

新增工具时，在 `app/tools/` 下实现一个类：

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
