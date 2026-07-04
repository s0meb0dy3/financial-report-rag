import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import {
  deleteDocument,
  deleteSession,
  getDocumentPage,
  getRuntimeConfig,
  getSession,
  listDocuments,
  listSessions,
  renameSession,
  streamChat,
  uploadDocument,
  type CitationResponse,
  type DocumentPageResponse,
  type DocumentResponse,
  type RuntimeConfigResponse,
  type SessionMessageResponse,
  type SessionSummaryResponse,
} from "./api/client";
import { AssistantMessage, UsageBar, UserMessage, type Message } from "./components/Messages";
import { DocumentPanel } from "./components/PdfViewer";
import { Sidebar } from "./components/Sidebar";
import "./styles.css";

function createSessionId() {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `session-${Math.random().toString(36).slice(2, 10)}`;
}

function makeId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

function mapMessage(message: SessionMessageResponse): Message {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    citations: message.citations ?? [],
    toolResults: message.tool_results ?? [],
    reasoning: message.reasoning_content ?? "",
    usage: message.usage ?? null,
  };
}

function App() {
  const [sessionId, setSessionId] = useState(createSessionId);
  const [sessions, setSessions] = useState<SessionSummaryResponse[]>([]);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("准备开始");
  const [isSending, setIsSending] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [documentPanelCollapsed, setDocumentPanelCollapsed] = useState(true);
  const [documentPanelWidth, setDocumentPanelWidth] = useState(420);
  const [isResizing, setIsResizing] = useState(false);
  const [activeDocumentId, setActiveDocumentId] = useState("");
  const [scrollToPage, setScrollToPage] = useState(0);
  const [visiblePage, setVisiblePage] = useState(1);
  const [pageDetail, setPageDetail] = useState<DocumentPageResponse | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfigResponse | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(() => input.trim().length > 0 && !isSending, [input, isSending]);
  const chatTitle = useMemo(() => {
    const session = sessions.find((s) => s.id === sessionId);
    return session?.title || "新对话";
  }, [sessions, sessionId]);
  const sessionUsage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].usage) return messages[i].usage;
    }
    return null;
  }, [messages]);

  const refreshSessions = useCallback(() => {
    listSessions()
      .then(setSessions)
      .catch((error: Error) => {
        setStatus(`历史列表加载失败：${error.message}`);
      });
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    getRuntimeConfig()
      .then(setRuntimeConfig)
      .catch((error: Error) => setStatus(`运行配置加载失败：${error.message}`));
  }, []);

  useEffect(() => {
    let cancelled = false;
    listDocuments()
      .then((items) => {
        if (cancelled) return;
        setDocuments(items);
        setActiveDocumentId((current) => current || items[0]?.id || "");
      })
      .catch((error: Error) => {
        if (!cancelled) setStatus(`报告列表加载失败：${error.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!isSending) return;
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [isSending]);

  useEffect(() => {
    let cancelled = false;
    setStatus("正在恢复会话");
    getSession(sessionId)
      .then((detail) => {
        if (cancelled) return;
        setMessages(detail.messages.map(mapMessage));
        setStatus(detail.messages.length ? "历史已恢复" : "准备开始");
      })
      .catch((error: Error) => {
        if (cancelled) return;
        if (error.message.includes("Session not found") || error.message.includes("404")) {
          setMessages([]);
          setStatus("准备开始");
          return;
        }
        setStatus(`加载失败：${error.message}`);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, status]);

  const updateAssistant = useCallback((assistantId: string, update: (m: Message) => Message) => {
    setMessages((items) => items.map((item) => (item.id === assistantId ? update(item) : item)));
  }, []);

  const switchSession = useCallback((id: string) => {
    setSessionId(id);
    setInput("");
    setStatus("准备开始");
  }, []);

  const startNewChat = useCallback(() => {
    const nextId = createSessionId();
    setSessionId(nextId);
    setMessages([]);
    setInput("");
    setStatus("准备开始");
  }, []);

  const openDocumentPage = useCallback((docId: string, page: number) => {
    if (!docId || page < 1) return;
    setActiveDocumentId(docId);
    setScrollToPage(page);
    setDocumentPanelCollapsed(false);
  }, []);

  const openCitation = useCallback(
    (citation: CitationResponse) => {
      if (!citation.page) return;
      openDocumentPage(citation.doc_id, citation.page);
    },
    [openDocumentPage],
  );

  const handleVisiblePageChange = useCallback(
    (page: number) => {
      setVisiblePage(page);
      if (activeDocumentId) {
        getDocumentPage(activeDocumentId, page)
          .then(setPageDetail)
          .catch((error: Error) => {
            setPageDetail(null);
            setStatus(`页面内容加载失败：${error.message}`);
          });
      }
    },
    [activeDocumentId],
  );

  const sendQuestion = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isSending) return;

      const startedAt = Date.now();
      const userMessage: Message = {
        id: makeId("user"),
        role: "user",
        content: trimmed,
        citations: [],
        toolResults: [],
        reasoning: "",
        usage: null,
      };
      const assistantId = makeId("assistant");
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
        toolResults: [],
        reasoning: "",
        usage: null,
        startedAt,
      };
      const controller = new AbortController();
      abortRef.current = controller;

      setInput("");
      setIsSending(true);
      setNow(startedAt);
      setStatus("提交问题");
      setMessages((items) => [...items, userMessage, assistantMessage]);

      try {
        await streamChat(
          { question: trimmed, session_id: sessionId, doc_id: activeDocumentId || null, visible_page: activeDocumentId ? visiblePage : null },
          (event) => {
            if (event.event === "status") {
              setStatus(event.data.message);
              return;
            }
            if (event.event === "usage") {
              updateAssistant(assistantId, (m) => ({ ...m, usage: event.data }));
              return;
            }
            if (event.event === "tool_call") {
              updateAssistant(assistantId, (m) => ({
                ...m,
                toolResults: [
                  ...m.toolResults.filter((item) => item.id !== event.data.id),
                  { ...event.data, status: "running" },
                ],
              }));
              setStatus("调用工具");
              return;
            }
            if (event.event === "tool_result") {
              updateAssistant(assistantId, (m) => ({
                ...m,
                toolResults: [
                  ...m.toolResults.filter((item) => item.id !== event.data.id),
                  event.data,
                ],
              }));
              setStatus(event.data.status === "error" ? "工具失败" : "工具完成");
              return;
            }
            if (event.event === "reasoning_delta") {
              updateAssistant(assistantId, (m) => ({
                ...m,
                reasoning: m.reasoning + event.data.content,
              }));
              return;
            }
            if (event.event === "answer_delta") {
              updateAssistant(assistantId, (m) => ({
                ...m,
                content: m.content + event.data.content,
              }));
              return;
            }
            if (event.event === "final") {
              updateAssistant(assistantId, (m) => ({
                ...m,
                content: event.data.answer,
                citations: event.data.citations,
                toolResults: event.data.tool_results ?? m.toolResults,
                reasoning: event.data.reasoning_content || m.reasoning,
                usage: event.data.usage ?? m.usage,
                completedAt: Date.now(),
              }));
              setStatus("完成");
              refreshSessions();
            }
          },
          controller.signal,
        );
      } catch (error) {
        const aborted = error instanceof DOMException && error.name === "AbortError";
        if (aborted) {
          setStatus("已停止生成");
          updateAssistant(assistantId, (item) => ({
            ...item,
            content: item.content || "已停止生成。",
            completedAt: Date.now(),
          }));
        } else {
          const msg = error instanceof Error ? error.message : "请求失败";
          setStatus(`出错：${msg}`);
          updateAssistant(assistantId, (item) => ({
            ...item,
            content: "请求失败，请稍后重试。",
            completedAt: Date.now(),
          }));
        }
      } finally {
        abortRef.current = null;
        setIsSending(false);
      }
    },
    [isSending, sessionId, activeDocumentId, visiblePage, updateAssistant, refreshSessions],
  );

  const submit = useCallback(
    async (event?: React.FormEvent) => {
      event?.preventDefault();
      await sendQuestion(input);
    },
    [input, sendQuestion],
  );

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const regenerateLast = useCallback(() => {
    if (isSending) return;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role !== "assistant") continue;
      const previous = messages[i - 1];
      if (previous?.role !== "user") return;
      void sendQuestion(previous.content);
      return;
    }
  }, [isSending, messages, sendQuestion]);

  const handleRenameSession = useCallback((id: string, title: string) => {
    renameSession(id, title)
      .then((updated) => {
        setSessions((items) => items.map((item) => (item.id === id ? updated : item)));
        setStatus("会话已重命名");
      })
      .catch((error: Error) => setStatus(`重命名失败：${error.message}`));
  }, []);

  const handleDeleteSession = useCallback((id: string) => {
    if (isSending) {
      setStatus("生成中，先停止再删除会话");
      return;
    }
    if (!window.confirm("确定删除这个会话？此操作不可撤销。")) return;
    deleteSession(id)
      .then(() => {
        setSessions((items) => items.filter((item) => item.id !== id));
        if (id === sessionId) startNewChat();
        setStatus("会话已删除");
      })
      .catch((error: Error) => setStatus(`删除失败：${error.message}`));
  }, [isSending, sessionId, startNewChat]);

  const handleUploadDocument = useCallback((file: File) => {
    uploadDocument(file)
      .then((doc) => {
        setDocuments((items) => [...items.filter((item) => item.id !== doc.id), doc].sort((a, b) => a.name.localeCompare(b.name)));
        setActiveDocumentId(doc.id);
        setDocumentPanelCollapsed(false);
        setStatus(doc.parsed ? "文档已上传并解析" : "PDF 已上传，等待解析后可问答");
      })
      .catch((error: Error) => setStatus(`上传失败：${error.message}`));
  }, []);

  const handleDeleteDocument = useCallback((docId: string) => {
    const doc = documents.find((item) => item.id === docId);
    if (!window.confirm(`确定删除 ${doc?.name ?? "这个文档"}？此操作不可撤销。`)) return;
    deleteDocument(docId)
      .then(() => {
        const remaining = documents.filter((item) => item.id !== docId);
        setDocuments(remaining);
        setActiveDocumentId((current) => current === docId ? remaining[0]?.id || "" : current);
        setPageDetail(null);
        setStatus("文档已删除");
      })
      .catch((error: Error) => setStatus(`文档删除失败：${error.message}`));
  }, [documents]);

  const askCurrentPage = useCallback(() => {
    const doc = documents.find((item) => item.id === activeDocumentId);
    if (!doc || !doc.parsed || isSending) return;
    void sendQuestion(`请基于《${doc.name}》第 ${visiblePage} 页内容，概括关键财务信息并指出值得关注的风险或变化。`);
  }, [activeDocumentId, documents, isSending, sendQuestion, visiblePage]);

  return (
    <main
      className={[
        "app-shell",
        sidebarCollapsed ? "sidebar-collapsed" : "",
        documentPanelCollapsed ? "document-collapsed" : "",
        isResizing ? "resizing" : "",
      ].filter(Boolean).join(" ")}
      style={{ "--doc-panel-width": `${documentPanelWidth}px` } as React.CSSProperties}
    >
      <Sidebar
        sessions={sessions}
        activeSessionId={sessionId}
        onNewChat={startNewChat}
        onSelectSession={switchSession}
        onRenameSession={handleRenameSession}
        onDeleteSession={handleDeleteSession}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((v) => !v)}
      />
      <section className="chat-main" aria-label="聊天">
        <header className="topbar">
          <div>
            <h1>{chatTitle}</h1>
            {runtimeConfig ? (
              <div className="runtime-config" aria-label="运行配置">
                <span>{runtimeConfig.status === "ok" ? "后端正常" : "后端异常"}</span>
                <span>{runtimeConfig.chat_model}</span>
                <span>{runtimeConfig.chat_base_url}</span>
                <span>{runtimeConfig.api_key_configured ? "Key 已配置" : "Key 未配置"}</span>
                <span>{runtimeConfig.mineru_api_key_configured ? "MinerU 已配置" : "MinerU 未配置"}</span>
              </div>
            ) : null}
          </div>
          <span className="runtime-status">{status}</span>
        </header>

        <div className="message-list" ref={listRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>开始对话</h2>
              <p>输入任何问题，开始聊天</p>
            </div>
          ) : (
            messages.map((message) =>
              message.role === "assistant" ? (
                <AssistantMessage
                  message={message}
                  now={now}
                  onOpenCitation={openCitation}
                  key={message.id}
                />
              ) : (
                <UserMessage content={message.content} key={message.id} />
              ),
            )
          )}
        </div>

        <div className="composer-dock">
          <UsageBar usage={sessionUsage} status={status} />
          <form className="composer" onSubmit={submit}>
            <textarea
              aria-label="输入问题"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submit();
                }
              }}
              placeholder="输入消息..."
              rows={1}
            />
            <div className="composer-actions">
              {isSending ? (
                <button type="button" className="secondary-action" onClick={stopGeneration}>
                  停止
                </button>
              ) : (
                <button type="button" className="secondary-action" disabled={!messages.some((m) => m.role === "assistant")} onClick={regenerateLast}>
                  再问
                </button>
              )}
              <button type="submit" disabled={!canSend}>
                {isSending ? "生成中" : "发送"}
              </button>
            </div>
          </form>
        </div>
      </section>
      <DocumentPanel
        documents={documents}
        activeDocId={activeDocumentId}
        scrollToPage={scrollToPage}
        pageDetail={pageDetail}
        collapsed={documentPanelCollapsed}
        onToggle={() => setDocumentPanelCollapsed((v) => !v)}
        onOpenPage={openDocumentPage}
        onVisiblePageChange={handleVisiblePageChange}
        onUploadDocument={handleUploadDocument}
        onDeleteDocument={handleDeleteDocument}
        onAskCurrentPage={askCurrentPage}
        onResize={setDocumentPanelWidth}
        onResizeStart={() => setIsResizing(true)}
        onResizeEnd={() => setIsResizing(false)}
      />
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
