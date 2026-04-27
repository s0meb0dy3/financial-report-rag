import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  ArrowRight,
  ArrowUp,
  BarChart3,
  Bot,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FileText,
  MessageSquarePlus,
  PanelRightOpen,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import {
  createSession as createBackendSession,
  deleteDocument as deleteBackendDocument,
  deleteSession as deleteBackendSession,
  getSession,
  listDocumentJobs,
  listDocuments,
  listSessions,
  streamChat,
  uploadDocument,
  updateSession as updateBackendSession,
  type CitationResponse,
  type DocumentJobResponse,
  type DocumentResponse,
  type SessionDetailResponse,
  type SessionMessageResponse,
  type SessionSummaryResponse,
  type ToolTraceResponse,
} from "./api/client";
import "./styles.css";

type Citation = {
  docId: string;
  docName: string;
  page: number | null;
};

type ToolTrace = {
  id: string;
  name: string;
  status: "done" | "idle" | "running";
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
  documents: string[];
  updatedAt: string;
  messages: Message[];
};

type DocumentRef = {
  id: string;
  name: string;
  chunkCount: number | null;
};

function makeId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}

function formatUpdatedAt(value: string) {
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return "刚刚";
  const diff = Math.max(0, Date.now() - timestamp);
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (diff < minute) return "刚刚";
  if (diff < hour) return `${Math.floor(diff / minute)} 分钟前`;
  if (diff < day) return `${Math.floor(diff / hour)} 小时前`;
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric" }).format(
    new Date(timestamp),
  );
}

function mapDocument(document: DocumentResponse): DocumentRef {
  return {
    id: document.doc_id,
    name: document.doc_name,
    chunkCount: document.chunk_count ?? null,
  };
}

function normalizeDocumentIds(documentIds: unknown, fallbackDocumentId: unknown = null) {
  const candidates = Array.isArray(documentIds)
    ? documentIds
    : typeof fallbackDocumentId === "string"
      ? [fallbackDocumentId]
      : [];
  const seen = new Set<string>();
  return candidates.filter((item): item is string => {
    if (typeof item !== "string") return false;
    const value = item.trim();
    if (!value || seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function validSessionDocuments(session: SessionSummaryResponse, documents: DocumentRef[]) {
  const available = new Set(documents.map((document) => document.id));
  return normalizeDocumentIds(session.doc_ids, session.doc_id).filter((docId) => available.has(docId));
}

function formatDocumentSelection(documentIds: string[], documents: DocumentRef[]) {
  if (!documentIds.length) return "未选择文档";
  const names = documentIds
    .map((docId) => documents.find((document) => document.id === docId)?.name)
    .filter((name): name is string => Boolean(name));
  if (!names.length) return `${documentIds.length} 份文档`;
  if (names.length === 1) return names[0];
  return `${names.slice(0, 2).join("、")}${names.length > 2 ? ` 等 ${names.length} 份` : ""}`;
}

function mapCitation(citation: CitationResponse): Citation {
  return {
    docId: citation.doc_id,
    docName: citation.doc_name,
    page: citation.page ?? null,
  };
}

function summarizeTool(tool: ToolTraceResponse) {
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

function mapTool(tool: ToolTraceResponse): ToolTrace {
  return {
    id: tool.tool_call_id || `${tool.tool_name}-${JSON.stringify(tool.arguments).slice(0, 24)}`,
    name: tool.tool_name,
    status: "done",
    detail: summarizeTool(tool),
  };
}

function statusTool(message: string): ToolTrace {
  return {
    id: `status-${message}`,
    name: "status",
    status: "running",
    detail: message,
  };
}

function cleanAssistantContent(content: string) {
  return normalizeMarkdownTables(content.replace(/^thought\s*\n+/i, "").trim());
}

type MarkdownSegment =
  | { type: "markdown"; content: string }
  | { type: "table"; rows: string[][] };

function normalizeMarkdownTables(content: string) {
  return content
    .split(/\r?\n/)
    .map((line) => {
      const pipeCount = (line.match(/\|/g) ?? []).length;
      if (line.trim().startsWith("|") && pipeCount >= 8) {
        return line.replace(/\|\s*\|\s*/g, "|\n| ");
      }
      return line;
    })
    .join("\n")
    .replace(/\n{3,}/g, "\n\n");
}

function isTableLine(line: string) {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.endsWith("|") && (trimmed.match(/\|/g) ?? []).length >= 2;
}

function splitTableRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isDividerRow(row: string[]) {
  return row.length > 0 && row.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")));
}

function parseMarkdownSegments(content: string): MarkdownSegment[] {
  const lines = normalizeMarkdownTables(content).split(/\r?\n/);
  const segments: MarkdownSegment[] = [];
  const markdownBuffer: string[] = [];

  function flushMarkdown() {
    const value = markdownBuffer.join("\n").trim();
    if (value) {
      segments.push({ type: "markdown", content: value });
    }
    markdownBuffer.length = 0;
  }

  for (let index = 0; index < lines.length; index += 1) {
    const currentLine = lines[index];
    const nextLine = lines[index + 1];
    if (isTableLine(currentLine) && nextLine && isDividerRow(splitTableRow(nextLine))) {
      flushMarkdown();
      const tableLines: string[] = [];
      while (index < lines.length && isTableLine(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      index -= 1;

      const rows = tableLines.map(splitTableRow);
      if (rows.length >= 2 && isDividerRow(rows[1])) {
        segments.push({ type: "table", rows: [rows[0], ...rows.slice(2)] });
      } else {
        markdownBuffer.push(...tableLines);
      }
      continue;
    }

    markdownBuffer.push(currentLine);
  }

  flushMarkdown();
  return segments;
}

function MarkdownTable({ rows }: { rows: string[][] }) {
  const [head, ...body] = rows;
  return (
    <div className="markdown-table-wrap">
      <table>
        <thead>
          <tr>
            {head.map((cell, index) => (
              <th key={`${cell}-${index}`}>
                <ReactMarkdown>{cell}</ReactMarkdown>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, rowIndex) => (
            <tr key={`${row.join("-")}-${rowIndex}`}>
              {head.map((_, cellIndex) => (
                <td key={`${rowIndex}-${cellIndex}`}>
                  <ReactMarkdown>{row[cellIndex] ?? ""}</ReactMarkdown>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <>
      {parseMarkdownSegments(content).map((segment, index) =>
        segment.type === "table" ? (
          <MarkdownTable key={`table-${index}`} rows={segment.rows} />
        ) : (
          <ReactMarkdown key={`markdown-${index}`}>{segment.content}</ReactMarkdown>
        ),
      )}
    </>
  );
}

function welcomeMessage(): Message {
  return {
    id: makeId("assistant"),
    role: "assistant",
    content: "上传或选择一份年报，然后开始提问。我会把答案、引用和检索路径放在同一个工作区里。",
  };
}

function mapSessionSummary(session: SessionSummaryResponse, documents: DocumentRef[]): Session {
  const selectedDocuments = validSessionDocuments(session, documents);
  return {
    id: session.id,
    title: session.title,
    documents: selectedDocuments,
    updatedAt: formatUpdatedAt(session.updated_at),
    messages: [welcomeMessage()],
  };
}

function mapSessionMessage(message: SessionMessageResponse): Message {
  const role = message.role === "user" ? "user" : "assistant";
  return {
    id: message.id,
    role,
    content: role === "assistant" ? cleanAssistantContent(message.content) : message.content,
    citations: message.citations?.map(mapCitation) ?? [],
    tools: message.tool_results?.map(mapTool) ?? [],
  };
}

function mergeSessionDetail(detail: SessionDetailResponse, documents: DocumentRef[]): Session {
  const messages = detail.messages?.map(mapSessionMessage) ?? [];
  return {
    ...mapSessionSummary(detail.session, documents),
    messages: messages.length ? messages : [welcomeMessage()],
  };
}

function errorMessage(error: unknown, fallback = "请求失败") {
  return error instanceof Error ? error.message : fallback;
}

function isActiveDocumentJob(job: DocumentJobResponse) {
  return job.status === "queued" || job.status === "running";
}

function jobStageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: "排队中",
    parsing: "解析 PDF",
    chunking: "生成 chunks",
    embedding: "生成向量",
    indexing: "写入索引",
    done: "已完成",
    failed: "失败",
  };
  return labels[stage] ?? stage;
}

function App() {
  const [documents, setDocuments] = useState<DocumentRef[]>([]);
  const [documentJobs, setDocumentJobs] = useState<DocumentJobResponse[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState("");
  const [draft, setDraft] = useState("");
  const [showInspector, setShowInspector] = useState(true);
  const [showDocumentManager, setShowDocumentManager] = useState(false);
  const [showDocumentPicker, setShowDocumentPicker] = useState(false);
  const [hasEntered, setHasEntered] = useState(false);
  const [apiStatus, setApiStatus] = useState<"loading" | "connected" | "error">("loading");
  const [apiError, setApiError] = useState("");
  const [documentError, setDocumentError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const draftRef = useRef<HTMLTextAreaElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);

  const refreshDocuments = useCallback(async () => {
    const documentPayload = await listDocuments();
    const loadedDocuments = documentPayload.documents.map(mapDocument);
    setDocuments(loadedDocuments);
    setSessions((current) =>
      current.map((session) => ({
        ...session,
        documents: session.documents.filter((docId) =>
          loadedDocuments.some((document) => document.id === docId),
        ),
      })),
    );
    return loadedDocuments;
  }, []);

  const refreshDocumentJobs = useCallback(async () => {
    const payload = await listDocumentJobs();
    setDocumentJobs(payload.jobs);
    return payload.jobs;
  }, []);

  const hydrateSession = useCallback(
    async (sessionId: string, availableDocuments = documents) => {
      const detail = await getSession(sessionId);
      setSessions((current) =>
        current.map((session) =>
          session.id === sessionId ? mergeSessionDetail(detail, availableDocuments) : session,
        ),
      );
    },
    [documents],
  );

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      try {
        setApiStatus("loading");
        const documentPayload = await listDocuments();
        if (cancelled) return;

        const loadedDocuments = documentPayload.documents.map(mapDocument);
        setDocuments(loadedDocuments);
        const jobsPayload = await listDocumentJobs();
        if (cancelled) return;
        setDocumentJobs(jobsPayload.jobs);

        const sessionPayload = await listSessions();
        let summaries = sessionPayload.sessions;
        if (!summaries.length) {
          const created = await createBackendSession({
            title: "新的财报对话",
            doc_ids: loadedDocuments[0] ? [loadedDocuments[0].id] : [],
          });
          summaries = [created];
        }
        if (cancelled) return;

        const nextSessions = summaries.map((session) => mapSessionSummary(session, loadedDocuments));
        setSessions(nextSessions);
        const firstSession = nextSessions[0];
        if (firstSession) {
          setActiveSessionId(firstSession.id);
          const detail = await getSession(firstSession.id);
          if (cancelled) return;
          setSessions((current) =>
            current.map((session) =>
              session.id === firstSession.id
                ? mergeSessionDetail(detail, loadedDocuments)
                : session,
            ),
          );
        }
        setApiStatus("connected");
        setApiError("");
      } catch (error) {
        if (cancelled) return;
        const message = errorMessage(error, "API 连接失败");
        const localSession: Session = {
          id: "local-preview",
          title: "FastAPI 未连接",
          documents: [],
          updatedAt: "刚刚",
          messages: [
            {
              id: "local-preview-message",
              role: "assistant",
              content: `暂时无法连接后端：${message}`,
              tools: [{ id: "api", name: "api", status: "idle", detail: "等待 FastAPI 可用" }],
            },
          ],
        };
        setSessions([localSession]);
        setActiveSessionId(localSession.id);
        setApiStatus("error");
        setApiError(message);
      }
    }

    void bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? sessions[0] ?? null,
    [activeSessionId, sessions],
  );

  const activeDocumentIds = activeSession
    ? activeSession.documents.filter((docId) => documents.some((document) => document.id === docId))
    : [];
  const activeDocuments = activeDocumentIds
    .map((docId) => documents.find((document) => document.id === docId))
    .filter((document): document is DocumentRef => Boolean(document));
  const activeDocumentLabel = formatDocumentSelection(activeDocumentIds, documents);
  const hasActiveJob = documentJobs.some(isActiveDocumentJob);
  const visibleMessages = activeSession?.messages.length ? activeSession.messages : [welcomeMessage()];
  const lastAssistant = [...visibleMessages].reverse().find((message) => message.role === "assistant");
  const citations = lastAssistant?.citations ?? [];
  const tools = lastAssistant?.tools?.length
    ? lastAssistant.tools
    : [{ id: "idle-search", name: "search_reports", status: "idle" as const, detail: "等待下一次检索" }];

  useEffect(() => {
    if (!hasEntered || !documentJobs.some(isActiveDocumentJob)) return;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const jobs = await refreshDocumentJobs();
          if (!jobs.some(isActiveDocumentJob)) {
            await refreshDocuments();
          }
          setApiStatus("connected");
          setDocumentError("");
        } catch (error) {
          setDocumentError(errorMessage(error, "文档状态刷新失败"));
        }
      })();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [documentJobs, hasEntered, refreshDocumentJobs, refreshDocuments]);

  async function selectSession(sessionId: string) {
    setActiveSessionId(sessionId);
    setDraft("");
    try {
      await hydrateSession(sessionId);
      setApiStatus("connected");
      setApiError("");
    } catch (error) {
      setApiStatus("error");
      setApiError(errorMessage(error));
    }
  }

  async function createSession() {
    try {
      const created = await createBackendSession({
        title: "新的财报对话",
        doc_ids: documents[0] ? [documents[0].id] : [],
      });
      const nextSession = mapSessionSummary(created, documents);
      setSessions((current) => [nextSession, ...current]);
      setActiveSessionId(nextSession.id);
      setDraft("");
      setApiStatus("connected");
      setApiError("");
    } catch (error) {
      setApiStatus("error");
      setApiError(errorMessage(error));
    }
  }

  async function deleteSession(id: string) {
    if (sessions.length <= 1) return;
    try {
      await deleteBackendSession(id);
      const remaining = sessions.filter((session) => session.id !== id);
      setSessions(remaining);
      if (id === activeSessionId && remaining[0]) {
        setActiveSessionId(remaining[0].id);
        await hydrateSession(remaining[0].id);
      }
      setApiStatus("connected");
      setApiError("");
    } catch (error) {
      setApiStatus("error");
      setApiError(errorMessage(error));
    }
  }

  async function updateDocuments(documentIds: string[]) {
    if (!activeSession) return;
    const sessionId = activeSession.id;
    const available = new Set(documents.map((document) => document.id));
    const nextDocumentIds = normalizeDocumentIds(documentIds).filter((docId) => available.has(docId));
    setSessions((current) =>
      current.map((session) =>
        session.id === sessionId ? { ...session, documents: nextDocumentIds, updatedAt: "刚刚" } : session,
      ),
    );
    try {
      const updated = await updateBackendSession(sessionId, {
        doc_id: nextDocumentIds[0] ?? null,
        doc_ids: nextDocumentIds,
      });
      setSessions((current) =>
        current.map((session) =>
          session.id === sessionId
            ? {
                ...session,
                title: updated.title,
                documents: validSessionDocuments(updated, documents),
                updatedAt: formatUpdatedAt(updated.updated_at),
              }
            : session,
        ),
      );
      setApiStatus("connected");
      setApiError("");
    } catch (error) {
      setApiStatus("error");
      setApiError(errorMessage(error));
    }
  }

  function toggleDocument(documentId: string) {
    const selected = new Set(activeDocumentIds);
    if (selected.has(documentId)) {
      selected.delete(documentId);
    } else {
      selected.add(documentId);
    }
    void updateDocuments(Array.from(selected));
  }

  async function sendMessage() {
    const session = activeSession;
    const question = (draftRef.current?.value ?? draft).trim();
    if (!session || !question || isSending || !activeDocumentIds.length) return;

    const sessionId = session.id;
    const userMessage: Message = { id: makeId("user"), role: "user", content: question };
    const assistantId = makeId("assistant");
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      tools: [statusTool("准备请求 FastAPI")],
    };
    const nextTitle = session.title === "新的财报对话" ? question.slice(0, 18) : session.title;

    setSessions((current) =>
      current.map((item) =>
        item.id === sessionId
          ? {
              ...item,
              title: nextTitle,
              updatedAt: "刚刚",
              messages: [...item.messages, userMessage, assistantMessage],
            }
          : item,
      ),
    );
    if (draftRef.current) {
      draftRef.current.value = "";
    }
    setDraft("");
    setIsSending(true);
    setApiError("");

    const updateAssistant = (updater: (message: Message) => Message) => {
      setSessions((current) =>
        current.map((item) =>
          item.id === sessionId
            ? {
                ...item,
                updatedAt: "刚刚",
                messages: item.messages.map((message) =>
                  message.id === assistantId ? updater(message) : message,
                ),
              }
            : item,
        ),
      );
    };

    try {
      await streamChat(
        {
          question,
          session_id: sessionId,
          top_k: 5,
          doc_id: activeDocumentIds[0] ?? null,
          doc_ids: activeDocumentIds,
          include_tool_results: true,
        },
        (event) => {
          if (event.event === "status") {
            updateAssistant((message) => {
              const completedTools = (message.tools ?? []).filter((tool) => tool.status === "done");
              return { ...message, tools: [statusTool(event.data.message), ...completedTools] };
            });
          }
          if (event.event === "tool_result") {
            updateAssistant((message) => {
              const nextTool = mapTool(event.data);
              const existingTools = (message.tools ?? []).filter((tool) => tool.id !== nextTool.id);
              return { ...message, tools: [...existingTools, nextTool] };
            });
          }
          if (event.event === "answer_delta") {
            updateAssistant((message) => ({
              ...message,
              content: `${message.content}${event.data.content}`,
            }));
          }
          if (event.event === "final") {
            updateAssistant((message) => ({
              ...message,
              content: cleanAssistantContent(event.data.answer || message.content || "没有返回答案。"),
              citations: event.data.citations?.map(mapCitation) ?? [],
              tools: event.data.tool_results?.map(mapTool) ?? message.tools ?? [],
            }));
          }
          if (event.event === "error") {
            setApiError(event.data.message || "流式请求失败");
          }
        },
      );
      setApiStatus("connected");
      setApiError("");
    } catch (error) {
      const message = errorMessage(error);
      setApiStatus("error");
      setApiError(message);
      updateAssistant(() => ({
        id: assistantId,
        role: "assistant",
        content: `请求 FastAPI 失败：${message}`,
        tools: [{ id: "api-error", name: "chat/stream", status: "idle", detail: "后端请求未完成" }],
      }));
    } finally {
      setIsSending(false);
    }
  }

  async function handleUploadFile(file: File | undefined) {
    if (!file) return;
    setShowDocumentManager(true);
    setIsUploading(true);
    setDocumentError("");
    try {
      const job = await uploadDocument(file);
      setDocumentJobs((current) => [job, ...current.filter((item) => item.job_id !== job.job_id)]);
      setApiStatus("connected");
    } catch (error) {
      setDocumentError(errorMessage(error, "上传失败"));
      setApiStatus("error");
    } finally {
      setIsUploading(false);
      if (uploadInputRef.current) {
        uploadInputRef.current.value = "";
      }
    }
  }

  async function deleteManagedDocument(documentId: string) {
    if (hasActiveJob) return;
    const document = documents.find((item) => item.id === documentId);
    const confirmed = window.confirm(`删除 ${document?.name ?? documentId} 的运行产物和索引？原 PDF 会保留。`);
    if (!confirmed) return;
    setDocumentError("");
    try {
      await deleteBackendDocument(documentId);
      const loadedDocuments = await refreshDocuments();
      const sessionPayload = await listSessions();
      setSessions((current) =>
        sessionPayload.sessions.map((session) => {
          const existing = current.find((item) => item.id === session.id);
          return {
            ...mapSessionSummary(session, loadedDocuments),
            messages: existing?.messages ?? [welcomeMessage()],
          };
        }),
      );
      setApiStatus("connected");
    } catch (error) {
      setDocumentError(errorMessage(error, "删除文档失败"));
      setApiStatus("error");
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
                <strong>{sessions.length}</strong>
                <span>saved sessions</span>
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

          <button className="new-chat" onClick={() => void createSession()}>
            <MessageSquarePlus size={17} />
            新建对话
          </button>

          <div className="sidebar-section">
            <div className="section-label">Sessions</div>
            <div className="session-list">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  className={`session-item group ${session.id === activeSession?.id ? "is-active" : ""}`}
                  onClick={() => void selectSession(session.id)}
                >
                  <span>
                    <strong>{session.title}</strong>
                    <small>
                      {session.updatedAt} ·{" "}
                      {formatDocumentSelection(session.documents, documents)}
                    </small>
                  </span>
                  {sessions.length > 1 && (
                    <Trash2
                      className="delete-icon"
                      size={15}
                      onClick={(event) => {
                        event.stopPropagation();
                        void deleteSession(session.id);
                      }}
                    />
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-footer">
            <Sparkles size={16} />
            <span>{apiStatus === "connected" ? "SQLite sessions" : "API pending"}</span>
          </div>
        </aside>

        <section className="workspace">
          <header className="topbar">
            <div>
              <p className="eyebrow">当前工作区</p>
              <h1>{activeSession?.title ?? "正在加载会话"}</h1>
            </div>
          <div className="topbar-actions">
              <div className="doc-picker">
                <FileText size={16} />
                <button
                  className="doc-picker-trigger"
                  type="button"
                  onClick={() => setShowDocumentPicker((value) => !value)}
                  disabled={!activeSession || !documents.length}
                >
                  <span>{documents.length ? activeDocumentLabel : "暂无文档"}</span>
                  <small>{activeDocumentIds.length ? `${activeDocumentIds.length} 份` : "未选择"}</small>
                  <ChevronDown size={15} />
                </button>
                {showDocumentPicker ? (
                  <div className="doc-picker-menu">
                    <div className="doc-picker-tools">
                      <button type="button" onClick={() => void updateDocuments(documents.map((item) => item.id))}>
                        全选
                      </button>
                      <button type="button" onClick={() => void updateDocuments([])}>
                        清空
                      </button>
                    </div>
                    <div className="doc-picker-list">
                      {documents.map((document) => {
                        const checked = activeDocumentIds.includes(document.id);
                        return (
                          <button
                            type="button"
                            className={`doc-picker-option ${checked ? "is-selected" : ""}`}
                            key={document.id}
                            onClick={() => toggleDocument(document.id)}
                          >
                            <span className="check-box">{checked ? <Check size={13} /> : null}</span>
                            <span>
                              <strong>{document.name}</strong>
                              <small>{document.chunkCount ?? "?"} chunks</small>
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ) : null}
              </div>
              <button
                className="icon-button"
                aria-label="管理文档"
                onClick={() => setShowDocumentManager(true)}
              >
                <UploadCloud size={18} />
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
            <span className="query-doc-filter">文档过滤：{activeDocumentLabel}</span>
            <span className="divider" />
            <span>{activeDocuments.reduce((sum, document) => sum + (document.chunkCount ?? 0), 0)} chunks</span>
            <span className="divider" />
            <span>Top K 预设：5</span>
            <span className="divider" />
            <span>{apiStatus === "connected" ? "API 已连接" : apiStatus === "loading" ? "API 连接中" : "API 异常"}</span>
            {hasActiveJob ? (
              <>
                <span className="divider" />
                <span>文档处理中</span>
              </>
            ) : null}
          </div>

          <div className="messages">
            {visibleMessages.map((message, index) => (
              <article
                key={message.id}
                className={`message ${message.role === "user" ? "is-user" : "is-assistant"}`}
                style={{ animationDelay: `${index * 60}ms` }}
              >
                <div className="avatar">{message.role === "user" ? "你" : <Bot size={17} />}</div>
                <div className="bubble">
                  {message.content ? <MarkdownContent content={message.content} /> : <p>正在生成...</p>}
                  {message.citations?.length ? (
                    <div className="citation-row">
                      {message.citations.map((citation) => (
                        <span key={`${citation.docId}-${citation.page ?? "none"}`}>
                          {citation.docName} · p.{citation.page ?? "?"}
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
                placeholder={
                  !activeDocumentIds.length
                    ? "请先上传并选择至少一份年报..."
                    : isSending
                      ? "正在流式接收回答..."
                      : "询问营收、利润、现金流、风险因素，或跨公司对比..."
                }
                rows={2}
                disabled={isSending || !activeSession || !activeDocumentIds.length}
              />
              <button
                className="send-button"
                type="submit"
                aria-label="发送问题"
                disabled={isSending || !activeSession || !activeDocumentIds.length}
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
                    <div className="source-item" key={`${citation.docId}-${citation.page ?? "none"}`}>
                      <FileText size={16} />
                      <span>
                        <strong>{citation.docName}</strong>
                        <small>page {citation.page ?? "?"}</small>
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
                  <div className="tool-step" key={tool.id}>
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
                <strong>{tools.filter((tool) => tool.status === "done").length}</strong>
                <span>tools</span>
              </div>
              <div>
                <strong>{citations.length}</strong>
                <span>refs</span>
              </div>
            </section>
          </div>
        </aside>
        {showDocumentManager ? (
          <div className="document-manager-shell" role="dialog" aria-modal="true">
            <div className="document-manager-backdrop" onClick={() => setShowDocumentManager(false)} />
            <section className="document-manager">
              <header className="document-manager-head">
                <div>
                  <p className="eyebrow">Documents</p>
                  <h2>文档管理</h2>
                </div>
                <button
                  className="icon-button"
                  aria-label="关闭文档管理"
                  onClick={() => setShowDocumentManager(false)}
                >
                  <X size={17} />
                </button>
              </header>

              <div className="upload-zone">
                <input
                  ref={uploadInputRef}
                  className="sr-only"
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(event) => void handleUploadFile(event.target.files?.[0])}
                />
                <button
                  className="upload-action"
                  onClick={() => uploadInputRef.current?.click()}
                  disabled={isUploading || hasActiveJob}
                >
                  <UploadCloud size={18} />
                  {isUploading ? "上传中" : hasActiveJob ? "处理中" : "上传 PDF"}
                </button>
                <button
                  className="icon-button"
                  aria-label="刷新文档状态"
                  onClick={() => {
                    void refreshDocuments();
                    void refreshDocumentJobs();
                  }}
                >
                  <RefreshCw size={17} />
                </button>
              </div>

              {documentError ? <div className="api-error">{documentError}</div> : null}

              <section className="document-panel-block">
                <div className="block-title">任务状态</div>
                <div className="job-list">
                  {documentJobs.length ? (
                    documentJobs.slice(0, 5).map((job) => (
                      <div className={`job-item ${job.status}`} key={job.job_id}>
                        <span className={`dot ${job.status === "succeeded" ? "done" : isActiveDocumentJob(job) ? "running" : "idle"}`} />
                        <span>
                          <strong>{job.file_name}</strong>
                          <small>
                            {job.status === "failed"
                              ? job.error || "处理失败"
                              : `${jobStageLabel(job.stage)} · ${job.doc_name ?? "等待生成文档"}`}
                          </small>
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="empty-copy">暂无处理任务。</p>
                  )}
                </div>
              </section>

              <section className="document-panel-block">
                <div className="block-title">已索引文档</div>
                <div className="managed-document-list">
                  {documents.length ? (
                    documents.map((document) => (
                      <div className="managed-document" key={document.id}>
                        <FileText size={17} />
                        <span>
                          <strong>{document.name}</strong>
                          <small>{document.chunkCount ?? "?"} chunks</small>
                        </span>
                        <button
                          className="managed-delete"
                          onClick={() => void deleteManagedDocument(document.id)}
                          disabled={hasActiveJob}
                          aria-label={`删除 ${document.name}`}
                        >
                          <Trash2 size={15} />
                        </button>
                      </div>
                    ))
                  ) : (
                    <p className="empty-copy">还没有索引文档。上传 PDF 后，完成解析即可在会话中选择。</p>
                  )}
                </div>
              </section>
            </section>
          </div>
        ) : null}
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
