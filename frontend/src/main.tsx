import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  getSession,
  listSessions,
  streamChat,
  type CitationResponse,
  type SessionMessageResponse,
  type SessionSummaryResponse,
  type UsageResponse,
} from "./api/client";
import "./styles.css";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: CitationResponse[];
  reasoning: string;
  usage: UsageResponse | null;
  startedAt?: number;
  completedAt?: number;
};

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
    reasoning: message.reasoning_content ?? "",
    usage: message.usage ?? null,
  };
}

function Icon({
  name,
  className = "",
}: {
  name:
    | "bolt"
    | "chevron"
    | "copy"
    | "menu"
    | "more"
    | "plus"
    | "search"
  className?: string;
}) {
  const paths = {
    bolt: "M13 2 4 14h7l-1 8 10-13h-7l1-7Z",
    chevron: "m6 9 6 6 6-6",
    copy: "M8 8h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2Zm-2 8H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
    menu: "M7 4h10M7 12h10M7 20h10",
    more: "M6 12h.01M12 12h.01M18 12h.01",
    plus: "M12 5v14M5 12h14",
    search: "m21 21-4.4-4.4M10.8 18a7.2 7.2 0 1 1 0-14.4 7.2 7.2 0 0 1 0 14.4Z",
  };
  return (
    <svg className={`icon ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}

function CitationList({ citations }: { citations: CitationResponse[] }) {
  if (!citations.length) return null;
  return (
    <div className="citations" aria-label="引用来源">
      {citations.map((citation, index) => (
        <span className="citation" key={`${citation.doc_id}-${citation.page}-${index}`}>
          {citation.doc_name || citation.doc_id}
          {citation.page ? ` p.${citation.page}` : ""}
        </span>
      ))}
    </div>
  );
}

async function copyText(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  document.body.removeChild(textarea);
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const disabled = !text.trim();

  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={async () => {
        if (disabled) return;
        await copyText(text);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1200);
      }}
    >
      <Icon name="copy" />
      <span>{copied ? "已复制" : "复制"}</span>
    </button>
  );
}

function UsageBar({ usage, status }: { usage: UsageResponse | null; status: string }) {
  const ratio = usage?.context_ratio ?? 0;
  const percent = Math.round(ratio * 100);
  const tone = ratio >= 0.85 ? "danger" : ratio >= 0.6 ? "warning" : "safe";
  const used = usage?.context_used_tokens ?? usage?.prompt_tokens ?? 0;
  const win = usage?.context_window_tokens ?? 0;
  const circumference = 2 * Math.PI * 7;
  const offset = circumference - (ratio * circumference);

  return (
    <div className={`usage-pill ${tone}`} aria-label="会话上下文占用">
      <svg className="usage-ring" viewBox="0 0 20 20" aria-hidden="true">
        <circle className="ring-bg" cx="10" cy="10" r="7" />
        <circle
          className="ring-fg"
          cx="10"
          cy="10"
          r="7"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <span className="usage-percent">{percent}%</span>
      <div className="usage-tooltip" role="tooltip">
        <div className="usage-tooltip-row">
          <span>上下文占用</span>
          <strong>{used.toLocaleString()} / {win ? win.toLocaleString() : "-"} tokens</strong>
        </div>
        {usage?.reasoning_tokens ? (
          <div className="usage-tooltip-row">
            <span>reasoning</span>
            <strong>{usage.reasoning_tokens.toLocaleString()}</strong>
          </div>
        ) : null}
        {usage?.cached_tokens ? (
          <div className="usage-tooltip-row">
            <span>cached</span>
            <strong>{usage.cached_tokens.toLocaleString()}</strong>
          </div>
        ) : null}
        {usage?.estimated ? <div className="usage-tooltip-tag">估算</div> : null}
        <div className="usage-tooltip-status">{status}</div>
      </div>
    </div>
  );
}

function ReasoningBlock({ message, now }: { message: Message; now: number }) {
  const [open, setOpen] = useState(true);
  const elapsed =
    message.startedAt && (message.completedAt || now)
      ? Math.max(1, Math.round(((message.completedAt || now) - message.startedAt) / 1000))
      : null;
  const isThinking = Boolean(message.startedAt && !message.completedAt && !message.content);

  if (!message.reasoning && !isThinking) return null;

  return (
    <section className="assistant-reasoning">
      <button className="reasoning-title" type="button" onClick={() => setOpen((v) => !v)}>
        <Icon name="bolt" className="reasoning-icon" />
        <span>
          {isThinking ? "思考中" : "已思考"}
          {elapsed ? `（用时 ${elapsed} 秒）` : ""}
        </span>
        <Icon name="chevron" className={open ? "chevron open" : "chevron"} />
      </button>
      {open && message.reasoning ? (
        <div className="reasoning-body">
          <ReactMarkdown>{message.reasoning}</ReactMarkdown>
        </div>
      ) : null}
    </section>
  );
}

function AssistantMessage({ message, now }: { message: Message; now: number }) {
  return (
    <article className="assistant-message">
      <div className="assistant-inner">
        <ReasoningBlock message={message} now={now} />
        <section className="assistant-answer" aria-label="回答">
          <div className="markdown-body">
            {message.content ? (
              <ReactMarkdown>{message.content}</ReactMarkdown>
            ) : (
              <span className="typing">正在生成...</span>
            )}
          </div>
          <CitationList citations={message.citations} />
        </section>
        <div className="answer-actions" aria-label="消息操作">
          <CopyButton text={message.content} label="复制回答" />
        </div>
      </div>
    </article>
  );
}

function UserMessage({ content }: { content: string }) {
  return (
    <article className="user-message">
      <div className="user-bubble">{content}</div>
      <div className="user-actions" aria-label="用户消息操作">
        <CopyButton text={content} label="复制问题" />
      </div>
    </article>
  );
}

function Sidebar({
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
  collapsed,
  onToggle,
}: {
  sessions: SessionSummaryResponse[];
  activeSessionId: string;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const groups = useMemo(() => {
    const now = new Date();
    const today = now.toISOString().slice(0, 10);
    const yesterday = new Date(now.getTime() - 86400000).toISOString().slice(0, 10);
    const result: { label: string; items: SessionSummaryResponse[] }[] = [];
    let currentLabel = "";
    let currentItems: SessionSummaryResponse[] = [];

    for (const session of sessions) {
      const date = session.updated_at.slice(0, 10);
      let label = date;
      if (date === today) label = "今天";
      else if (date === yesterday) label = "昨天";

      if (label !== currentLabel) {
        if (currentItems.length) result.push({ label: currentLabel, items: currentItems });
        currentLabel = label;
        currentItems = [];
      }
      currentItems.push(session);
    }
    if (currentItems.length) result.push({ label: currentLabel, items: currentItems });
    return result;
  }, [sessions]);

  return (
    <aside className={collapsed ? "sidebar collapsed" : "sidebar"} aria-label="会话导航">
      <div className="brand-row">
        <div className="brand-mark">F</div>
        <span>fintell</span>
        <button type="button" aria-label="搜索">
          <Icon name="search" />
        </button>
        <button type="button" aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"} onClick={onToggle}>
          <Icon name="menu" />
        </button>
      </div>
      <button className="new-chat" type="button" onClick={onNewChat}>
        <Icon name="plus" />
        <span>开启新对话</span>
      </button>
      <nav className="history-list" aria-label="历史对话">
        {groups.map((group) => (
          <div className="history-group" key={group.label}>
            <p>{group.label}</p>
            {group.items.map((session) => (
              <button
                className={`history-item${session.id === activeSessionId ? " active" : ""}`}
                type="button"
                key={session.id}
                onClick={() => onSelectSession(session.id)}
              >
                <span>{session.title}</span>
                <Icon name="more" />
              </button>
            ))}
          </div>
        ))}
        {sessions.length === 0 && !collapsed && (
          <p className="history-empty">暂无对话</p>
        )}
      </nav>
    </aside>
  );
}

function App() {
  const [sessionId, setSessionId] = useState(createSessionId);
  const [sessions, setSessions] = useState<SessionSummaryResponse[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("准备开始");
  const [isSending, setIsSending] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
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
    listSessions().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

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
        reasoning: "",
        usage: null,
      };
      const assistantId = makeId("assistant");
      const assistantMessage: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        citations: [],
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
        await streamChat({ question, session_id: sessionId }, (event) => {
          if (event.event === "status") {
            setStatus(event.data.message);
            return;
          }
          if (event.event === "usage") {
            updateAssistant(assistantId, (m) => ({ ...m, usage: event.data }));
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
    [input, isSending, sessionId, updateAssistant, refreshSessions],
  );

  return (
    <main className={sidebarCollapsed ? "app-shell sidebar-collapsed" : "app-shell"}>
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
                <AssistantMessage message={message} now={now} key={message.id} />
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
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
