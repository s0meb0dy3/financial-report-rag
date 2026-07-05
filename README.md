# Fintell

极简前后端分离财报 Chatbox。后端负责 OpenAI-compatible 聊天、流式输出、工具调用、本地财报页读取、MinerU 精准解析上传和 SQLite 会话历史；前端负责单页对话、图表、PDF 预览和文档管理。当前版本不包含向量索引。

## 功能

- Chat：调用 OpenAI-compatible chat model，支持流式最终回答。
- Reasoning：兼容 MiMo 的 `reasoning_content` 流式展示和历史回传。
- Tools：轻量工具抽象层，内置本地财报读取、全文检索和图表生成，可选接入 Tavily web search。
- Documents：支持上传 PDF；配置 `MINERU_API_KEY` 时走 MinerU 精准解析，否则用本地 PyMuPDF 文本提取兜底。
- History：SQLite 持久化会话，刷新页面后可恢复历史。
- Usage：展示 token/context 占用，前端用 10 格占用条表达上下文压力。
- Frontend：金融风格单页 chatbox，支持 Markdown、GFM 表格、ECharts 图表和 PDF 预览。

## 环境变量

复制 `.env.example` 后填入密钥：

```bash
cp .env.example .env
```

最小配置：

```bash
CHAT_API_KEY=your-deepseek-api-key
CHAT_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-v4-flash
CHAT_THINKING_ENABLED=false
CHAT_PASS_REASONING_HISTORY=false
CHAT_STREAM_INCLUDE_USAGE=true
CONTEXT_WINDOW_TOKENS=128000
SESSION_DB_PATH=data/sessions.sqlite3
```

可选工具：

```bash
TAVILY_API_KEY=your-tavily-api-key
MINERU_API_KEY=your-mineru-api-key
```

默认使用 DeepSeek 的 OpenAI-compatible API。切换到其他兼容模型时，只改 `CHAT_API_KEY`、`CHAT_BASE_URL` 和 `CHAT_MODEL`。`CHAT_*` 优先于旧的 `MIMO_*`/`OPENROUTER_*`/`OPENAI_*` 配置；只设置旧 provider key 时仍保留对应兜底行为。MiMo thinking 模式下，历史 assistant 的 `reasoning_content` 会随 SQLite 历史一起回传给后续请求。

上传 PDF 时，如果配置了 `MINERU_API_KEY`，后端会调用 MinerU API 申请上传 URL、上传原 PDF、轮询解析结果、下载 zip，并归一化成项目内部使用的 `content_list_v2.json`。超过 200 页的 PDF 会按 `page_ranges` 自动切成每 200 页一段提交给 MinerU，读取时仍按原始全局页码访问。

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

真实模型 smoke check：

```bash
uv run python main.py chat "用一句话介绍你自己" --session-id smoke
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
- `GET /runtime/config`：返回模型和 MinerU key 是否已配置，不返回密钥明文。
- `GET /sessions`：列出已有会话。
- `GET /sessions/{session_id}`：恢复单个会话历史。
- `GET /documents`：列出已解析的本地财报。
- `POST /documents?filename=report.pdf`：上传 PDF；配置 MinerU 时自动精准解析。
- `DELETE /documents/{doc_id}`：删除上传的 PDF 和解析产物。
- `GET /documents/{doc_id}/toc`：读取 PDF 内置目录，返回可跳转的物理页码。
- `GET /documents/{doc_id}/pdf`：预览原始 PDF。
- `GET /documents/{doc_id}/pages/{page}`：读取指定 PDF 页文本。
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
- `app/documents/`：读取本地 PDF 和 MinerU 页级解析结果。
- `app/session/`：SQLite session 和 turn 历史。
- `app/tools/`：OpenAI-compatible tool 抽象，包含本地报告读取、图表生成和 Tavily web search。
- `frontend/`：React 单页 chatbox。
- `data/raw/`：原始财报 PDF，PDF 预览和来源定位会读取。

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
