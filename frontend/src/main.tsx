import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import {
  getDocumentPage,
  getSession,
  listDocuments,
  listSessions,
  streamChat,
  type CitationResponse,
  type DocumentPageResponse,
  type DocumentResponse,
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

  const submit = useCallback(
    async (event?: React.FormEvent) => {
      event?.preventDefault();
      const question = input.trim();
      if (!question || isSending) return;

      const startedAt = Date.now();
      const userMessage: Message = {
        id: makeId("user"),
        role: "user",
        content: question,
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

      setInput("");
      setIsSending(true);
      setNow(startedAt);
      setStatus("提交问题");
      setMessages((items) => [...items, userMessage, assistantMessage]);

      try {
        await streamChat({ question, session_id: sessionId, doc_id: activeDocumentId || null, visible_page: activeDocumentId ? visiblePage : null }, (event) => {
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
        });
      } catch (error) {
        const msg = error instanceof Error ? error.message : "请求失败";
        setStatus(`出错：${msg}`);
        updateAssistant(assistantId, (item) => ({
          ...item,
          content: "请求失败，请稍后重试。",
          completedAt: Date.now(),
        }));
      } finally {
        setIsSending(false);
      }
    },
    [input, isSending, sessionId, activeDocumentId, visiblePage, updateAssistant, refreshSessions],
  );

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
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((v) => !v)}
      />
      <section className="chat-main" aria-label="聊天">
        <header className="topbar">
          <div>
            <h1>{chatTitle}</h1>
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
            <button type="submit" disabled={!canSend}>
              {isSending ? "生成中" : "发送"}
            </button>
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
