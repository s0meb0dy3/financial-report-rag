import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  ArrowRight,
  ArrowUp,
  BarChart3,
  Bot,
  ChevronLeft,
  ChevronRight,
  FileText,
  MessageSquarePlus,
  PanelRightOpen,
  Search,
  Settings2,
  Sparkles,
  Trash2,
} from "lucide-react";
import "./styles.css";

type Citation = {
  docId: string;
  docName: string;
  page: number | null;
};

type ToolTrace = {
  name: string;
  status: "done" | "idle";
  detail: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  tools?: ToolTrace[];
};

type Session = {
  id: string;
  title: string;
  document: string;
  updatedAt: string;
  messages: Message[];
};

type DocumentRef = {
  id: string;
  name: string;
};

type ApiCitation = {
  doc_id: string;
  doc_name: string;
  page: number | null;
};

type ApiToolTrace = {
  tool_name: string;
  arguments: Record<string, unknown>;
  output: Record<string, unknown>;
  tool_call_id: string;
};

type ChatApiResponse = {
  answer: string;
  citations: ApiCitation[];
  tool_results: ApiToolTrace[];
};

const fallbackDocuments: DocumentRef[] = [
  { id: "moutai", name: "贵州茅台 2024 年报" },
  { id: "pingan", name: "中国平安 2024 年报" },
  { id: "cmb", name: "招商银行 2024 年报" },
];

const initialSessions: Session[] = [
  {
    id: "s1",
    title: "贵州茅台营收分析",
    document: "moutai",
    updatedAt: "刚刚",
    messages: [
      {
        id: "m1",
        role: "user",
        content: "贵州茅台 2024 年营业总收入是多少？请给出引用。",
      },
      {
        id: "m2",
        role: "assistant",
        content:
          "贵州茅台 2024 年营业总收入为 **1741.44 亿元**。该数据来自主要会计数据表，口径为合并报表营业总收入。",
        citations: [
          { docId: "moutai", docName: "贵州茅台 2024 年报", page: 12 },
          { docId: "moutai", docName: "贵州茅台 2024 年报", page: 86 },
        ],
        tools: [
          { name: "search_tables", status: "done", detail: "命中主要会计数据与利润表候选表" },
          { name: "extract_table", status: "done", detail: "读取完整表格并校验年份列" },
        ],
      },
    ],
  },
  {
    id: "s2",
    title: "现金流质量",
    document: "moutai",
    updatedAt: "12 分钟前",
    messages: [
      {
        id: "m3",
        role: "assistant",
        content:
          "可以从经营活动现金流净额、净利润匹配度和应收项目变化三个角度看现金流质量。",
        citations: [{ docId: "moutai", docName: "贵州茅台 2024 年报", page: 91 }],
      },
    ],
  },
  {
    id: "s3",
    title: "风险因素摘要",
    document: "pingan",
    updatedAt: "昨天",
    messages: [
      {
        id: "m4",
        role: "assistant",
        content: "当前摘要尚未生成。可以选择文档后继续追问。",
      },
    ],
  },
];

function makeId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

function summarizeTool(tool: ApiToolTrace) {
  const results = Array.isArray(tool.output.results) ? tool.output.results.length : undefined;
  const tables = Array.isArray(tool.output.tables) ? tool.output.tables.length : undefined;
  if (typeof results === "number") {
    return `返回 ${results} 条文本证据`;
  }
  if (typeof tables === "number") {
    return `返回 ${tables} 张候选表`;
  }
  if (tool.output.table) {
    return "读取完整表格";
  }
  return "已完成";
}

function cleanAssistantContent(content: string) {
  return content.replace(/^thought\s*\n+/i, "").trim();
}

function App() {
  const [documents, setDocuments] = useState<DocumentRef[]>(fallbackDocuments);
  const [sessions, setSessions] = useState(initialSessions);
  const [activeSessionId, setActiveSessionId] = useState(initialSessions[0].id);
  const [draft, setDraft] = useState("");
  const [showInspector, setShowInspector] = useState(true);
  const [hasEntered, setHasEntered] = useState(false);
  const [apiStatus, setApiStatus] = useState<"loading" | "connected" | "error">("loading");
  const [apiError, setApiError] = useState("");
  const [isSending, setIsSending] = useState(false);
  const draftRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadDocuments() {
      try {
        const response = await fetch("/api/documents");
        if (!response.ok) {
          throw new Error(`GET /documents ${response.status}`);
        }
        const payload = (await response.json()) as {
          documents?: Array<{ doc_id: string; doc_name: string }>;
        };
        const loadedDocuments = (payload.documents ?? []).map((document) => ({
          id: document.doc_id,
          name: document.doc_name,
        }));
        if (cancelled || !loadedDocuments.length) return;
        setDocuments(loadedDocuments);
        setSessions((current) =>
          current.map((session) =>
            loadedDocuments.some((document) => document.id === session.document)
              ? session
              : { ...session, document: loadedDocuments[0].id },
          ),
        );
        setApiStatus("connected");
        setApiError("");
      } catch (error) {
        if (cancelled) return;
        setApiStatus("error");
        setApiError(error instanceof Error ? error.message : "API 连接失败");
      }
    }

    void loadDocuments();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? sessions[0],
    [activeSessionId, sessions],
  );

  const activeDocumentId = documents.some((document) => document.id === activeSession.document)
    ? activeSession.document
    : documents[0]?.id;
  const activeDocument = documents.find((document) => document.id === activeDocumentId);
  const lastAssistant = [...activeSession.messages].reverse().find((message) => message.role === "assistant");
  const citations = lastAssistant?.citations ?? [];
  const tools = lastAssistant?.tools ?? [
    { name: "search_reports", status: "idle" as const, detail: "等待下一次检索" },
  ];

  function createSession() {
    const nextSession: Session = {
      id: makeId("session"),
      title: "新的财报对话",
      document: documents[0]?.id ?? fallbackDocuments[0].id,
      updatedAt: "刚刚",
      messages: [
        {
          id: makeId("assistant"),
          role: "assistant",
          content: "选择一份年报，然后开始提问。我会把答案、引用和检索路径放在同一个工作区里。",
        },
      ],
    };
    setSessions((current) => [nextSession, ...current]);
    setActiveSessionId(nextSession.id);
    setDraft("");
  }

  function deleteSession(id: string) {
    setSessions((current) => {
      const next = current.filter((session) => session.id !== id);
      if (id === activeSessionId && next[0]) {
        setActiveSessionId(next[0].id);
      }
      return next.length ? next : current;
    });
  }

  function updateDocument(documentId: string) {
    setSessions((current) =>
      current.map((session) =>
        session.id === activeSession.id ? { ...session, document: documentId, updatedAt: "刚刚" } : session,
      ),
    );
  }

  async function sendMessage() {
    const question = (draftRef.current?.value ?? draft).trim();
    if (!question || isSending) return;
    const userMessage: Message = { id: makeId("user"), role: "user", content: question };
    const nextTitle = activeSession.title === "新的财报对话" ? question.slice(0, 18) : activeSession.title;

    setSessions((current) =>
      current.map((session) =>
        session.id === activeSession.id
          ? {
              ...session,
              title: nextTitle,
              updatedAt: "刚刚",
              messages: [...session.messages, userMessage],
            }
          : session,
      ),
    );
    if (draftRef.current) {
      draftRef.current.value = "";
    }
    setDraft("");
    setIsSending(true);
    setApiError("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          session_id: activeSession.id,
          top_k: 5,
          doc_id: activeDocumentId,
          include_tool_results: true,
        }),
      });
      if (!response.ok) {
        throw new Error(`POST /chat ${response.status}`);
      }
      const payload = (await response.json()) as ChatApiResponse;
      const assistantMessage: Message = {
        id: makeId("assistant"),
        role: "assistant",
        content: cleanAssistantContent(payload.answer || "没有返回答案。"),
        citations: payload.citations.map((citation) => ({
          docId: citation.doc_id,
          docName: citation.doc_name,
          page: citation.page,
        })),
        tools: payload.tool_results.map((tool) => ({
          name: tool.tool_name,
          status: "done",
          detail: summarizeTool(tool),
        })),
      };
      setSessions((current) =>
        current.map((session) =>
          session.id === activeSession.id
            ? { ...session, updatedAt: "刚刚", messages: [...session.messages, assistantMessage] }
            : session,
        ),
      );
      setApiStatus("connected");
    } catch (error) {
      const message = error instanceof Error ? error.message : "请求失败";
      setApiStatus("error");
      setApiError(message);
      setSessions((current) =>
        current.map((session) =>
          session.id === activeSession.id
            ? {
                ...session,
                messages: [
                  ...session.messages,
                  {
                    id: makeId("assistant"),
                    role: "assistant",
                    content: `请求 FastAPI 失败：${message}`,
                    tools: [{ name: "api/chat", status: "idle", detail: "后端请求未完成" }],
                  },
                ],
              }
            : session,
        ),
      );
    } finally {
      setIsSending(false);
    }
  }

  if (!hasEntered) {
    return (
      <main className="welcome">
        <div className="welcome-shell">
          <section className="welcome-copy">
            <div className="welcome-brand">
              <div className="welcome-mark">
                <BarChart3 size={19} />
              </div>
              <span>CaibaoAgent</span>
            </div>
            <p className="welcome-kicker">Financial report RAG</p>
            <h1>把年报问答，放回清晰的证据里。</h1>
            <p className="welcome-lede">
              面向财报检索、表格指标和引用追踪的轻量研究工作台。进入后可以新建会话、切换文档，并查看每次回答的来源。
            </p>
            <button className="enter-button" onClick={() => setHasEntered(true)}>
              进入工作台
              <ArrowRight size={18} />
            </button>
            <div className="welcome-meta">
              <div>
                <strong>{documents.length}</strong>
                <span>indexed reports</span>
              </div>
              <div>
                <strong>5</strong>
                <span>retrieval top k</span>
              </div>
              <div>
                <strong>2</strong>
                <span>evidence tools</span>
              </div>
            </div>
          </section>

          <section className="welcome-stage" aria-hidden="true">
            <div className="report-sheet secondary">
              <div className="sheet-head">
                <div className="sheet-title">Cash Flow</div>
                <div className="sheet-tag">verified</div>
              </div>
              {Array.from({ length: 7 }).map((_, index) => (
                <div className="sheet-row" key={`secondary-${index}`}>
                  <span />
                  <span />
                  <span />
                </div>
              ))}
              <div className="sheet-chart">
                <span className="h-14" />
                <span />
                <span />
                <span />
                <span />
              </div>
            </div>
            <div className="report-sheet">
              <div className="sheet-head">
                <div className="sheet-title">Annual Report</div>
                <div className="sheet-tag">p.12</div>
              </div>
              {Array.from({ length: 8 }).map((_, index) => (
                <div className="sheet-row" key={`primary-${index}`}>
                  <span />
                  <span />
                  <span />
                </div>
              ))}
              <div className="sheet-chart">
                <span className="h-16" />
                <span />
                <span />
                <span />
                <span />
              </div>
            </div>
          </section>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-paper text-ink">
      <div
        className={`grid min-h-screen grid-cols-[288px_minmax(0,1fr)] transition-[grid-template-columns] duration-300 ${
          showInspector
            ? "lg:grid-cols-[304px_minmax(0,1fr)_360px]"
            : "lg:grid-cols-[304px_minmax(0,1fr)_40px]"
        }`}
      >
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">
              <BarChart3 size={18} />
            </div>
            <div>
              <div className="brand-name">CaibaoAgent</div>
              <div className="brand-caption">Financial report RAG</div>
            </div>
          </div>

          <button className="new-chat" onClick={createSession}>
            <MessageSquarePlus size={17} />
            新建对话
          </button>

          <div className="sidebar-section">
            <div className="section-label">Sessions</div>
            <div className="session-list">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  className={`session-item group ${session.id === activeSession.id ? "is-active" : ""}`}
                  onClick={() => setActiveSessionId(session.id)}
                >
                  <span>
                    <strong>{session.title}</strong>
                    <small>{session.updatedAt} · {documents.find((doc) => doc.id === session.document)?.name}</small>
                  </span>
                  {sessions.length > 1 && (
                    <Trash2
                      className="delete-icon"
                      size={15}
                      onClick={(event) => {
                        event.stopPropagation();
                        deleteSession(session.id);
                      }}
                    />
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-footer">
            <Sparkles size={16} />
            <span>Prototype mode</span>
          </div>
        </aside>

        <section className="workspace">
          <header className="topbar">
            <div>
              <p className="eyebrow">当前工作区</p>
              <h1>{activeSession.title}</h1>
            </div>
            <div className="topbar-actions">
              <label className="doc-select">
                <FileText size={16} />
                <select value={activeDocumentId} onChange={(event) => updateDocument(event.target.value)}>
                  {documents.map((document) => (
                    <option key={document.id} value={document.id}>
                      {document.name}
                    </option>
                  ))}
                </select>
              </label>
              <button className="icon-button" aria-label="打开设置">
                <Settings2 size={18} />
              </button>
              <button
                className="icon-button lg:hidden"
                aria-label="切换证据面板"
                onClick={() => setShowInspector((value) => !value)}
              >
                <PanelRightOpen size={18} />
              </button>
            </div>
          </header>

          <div className="query-strip">
            <Search size={17} />
            <span>文档过滤：{activeDocument?.name}</span>
            <span className="divider" />
            <span>Top K 预设：5</span>
            <span className="divider" />
            <span>{apiStatus === "connected" ? "API 已连接" : apiStatus === "loading" ? "API 连接中" : "API 异常"}</span>
          </div>

          <div className="messages">
            {activeSession.messages.map((message, index) => (
              <article
                key={message.id}
                className={`message ${message.role === "user" ? "is-user" : "is-assistant"}`}
                style={{ animationDelay: `${index * 60}ms` }}
              >
                <div className="avatar">{message.role === "user" ? "你" : <Bot size={17} />}</div>
                <div className="bubble">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                  {message.citations?.length ? (
                    <div className="citation-row">
                      {message.citations.map((citation) => (
                        <span key={`${citation.docId}-${citation.page}`}>
                          {citation.docName} · p.{citation.page}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </article>
            ))}
          </div>

          <div className="composer-wrap">
            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                void sendMessage();
              }}
            >
              <textarea
                ref={draftRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
                placeholder={isSending ? "正在等待 FastAPI 返回..." : "询问营收、利润、现金流、风险因素或指定页码证据..."}
                rows={2}
                disabled={isSending}
              />
              <button
                className="send-button"
                type="submit"
                aria-label="发送问题"
                disabled={isSending}
              >
                <ArrowUp size={18} />
              </button>
            </form>
            {apiError ? <div className="api-error">{apiError}</div> : null}
          </div>
        </section>

        <aside className={`inspector ${showInspector ? "is-open" : "is-collapsed"}`}>
          <button
            className="collapse-button"
            aria-label={showInspector ? "折叠证据面板" : "展开证据面板"}
            onClick={() => setShowInspector((value) => !value)}
          >
            <span className="sr-only">{showInspector ? "折叠证据面板" : "展开证据面板"}</span>
            {showInspector ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>

          <div className="inspector-content">
            <div className="inspector-head">
              <p className="eyebrow">Evidence</p>
              <h2>引用与检索路径</h2>
            </div>

            <section className="inspector-block">
              <div className="block-title">引用来源</div>
              <div className="source-list">
                {citations.length ? (
                  citations.map((citation) => (
                    <div className="source-item" key={`${citation.docId}-${citation.page}`}>
                      <FileText size={16} />
                      <span>
                        <strong>{citation.docName}</strong>
                        <small>page {citation.page}</small>
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="empty-copy">暂无引用。</p>
                )}
              </div>
            </section>

            <section className="inspector-block">
              <div className="block-title">工具轨迹</div>
              <div className="timeline">
                {tools.map((tool) => (
                  <div className="tool-step" key={tool.name}>
                    <span className={`dot ${tool.status}`} />
                    <span>
                      <strong>{tool.name}</strong>
                      <small>{tool.detail}</small>
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="metric-band">
              <div>
                <strong>5</strong>
                <span>Top K</span>
              </div>
              <div>
                <strong>2</strong>
                <span>tools</span>
              </div>
              <div>
                <strong>{citations.length}</strong>
                <span>refs</span>
              </div>
            </section>
          </div>
        </aside>
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
