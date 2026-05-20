import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import ReactMarkdown from "react-markdown";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import remarkGfm from "remark-gfm";
import {
  documentPdfUrl,
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
  type ToolResultResponse,
  type UsageResponse,
} from "./api/client";
import "./styles.css";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: CitationResponse[];
  toolResults: ToolResultResponse[];
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
    toolResults: message.tool_results ?? [],
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
    | "file"
    | "menu"
    | "panel"
    | "plus"
  className?: string;
}) {
  const paths = {
    bolt: "M13 2 4 14h7l-1 8 10-13h-7l1-7Z",
    chevron: "m6 9 6 6 6-6",
    copy: "M8 8h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V10a2 2 0 0 1 2-2Zm-2 8H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1",
    file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Zm0 0v6h6M8 13h8M8 17h5",
    menu: "M7 4h10M7 12h10M7 20h10",
    panel: "M4 5h16v14H4zM14 5v14",
    plus: "M12 5v14M5 12h14",
  };
  return (
    <svg className={`icon ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}

function CitationList({
  citations,
  onOpenCitation,
}: {
  citations: CitationResponse[];
  onOpenCitation?: (citation: CitationResponse) => void;
}) {
  if (!citations.length) return null;
  return (
    <div className="citations" aria-label="引用来源">
      {citations.map((citation, index) => (
        <button
          className="citation"
          disabled={!citation.page || !onOpenCitation}
          key={`${citation.doc_id}-${citation.page}-${index}`}
          type="button"
          onClick={() => onOpenCitation?.(citation)}
        >
          {citation.doc_name || citation.doc_id}
          {citation.page ? ` p.${citation.page}` : ""}
        </button>
      ))}
    </div>
  );
}

function normalizeMarkdown(content: string) {
  return content
    .split("\n")
    .map((line) => {
      if (!line.includes("| |")) return line;
      const normalized = line.replace(/\|\s+\|/g, "|\n|");
      const rows = normalized.split("\n");
      const hasSeparator = rows.some((row) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(row));
      return hasSeparator ? normalized : line;
    })
    .join("\n");
}

function MarkdownContent({ content }: { content: string }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]}>{normalizeMarkdown(content)}</ReactMarkdown>;
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
  const activeCells = Math.min(10, Math.max(0, Math.ceil(ratio * 10)));

  return (
    <div className={`usage-pill ${tone}`} aria-label="会话上下文占用">
      <div className="usage-cells" aria-hidden="true">
        {Array.from({ length: 10 }).map((_, index) => (
          <span className={index < activeCells ? "active" : ""} key={index} />
        ))}
      </div>
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
          <MarkdownContent content={message.reasoning} />
        </div>
      ) : null}
    </section>
  );
}

function AssistantMessage({
  message,
  now,
  onOpenCitation,
}: {
  message: Message;
  now: number;
  onOpenCitation: (citation: CitationResponse) => void;
}) {
  return (
    <article className="assistant-message">
      <div className="assistant-inner">
        <ReasoningBlock message={message} now={now} />
        <ToolResults results={message.toolResults} />
        <section className="assistant-answer" aria-label="回答">
          <div className="markdown-body">
            {message.content ? (
              <MarkdownContent content={message.content} />
            ) : (
              <span className="typing">正在生成...</span>
            )}
          </div>
          <CitationList citations={message.citations} onOpenCitation={onOpenCitation} />
        </section>
        <div className="answer-actions" aria-label="消息操作">
          <CopyButton text={message.content} label="复制回答" />
        </div>
      </div>
    </article>
  );
}

const PAGE_GAP = 12;
const BUFFER_PAGES = 2;

function PdfScrollViewer({
  docId,
  scrollToPage,
  onVisiblePageChange,
}: {
  docId: string;
  scrollToPage: number;
  onVisiblePageChange: (page: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [totalPages, setTotalPages] = useState(0);
  const [pageHeights, setPageHeights] = useState<number[]>([]);
  const [renderRange, setRenderRange] = useState({ start: 0, end: 0 });
  const scaleRef = useRef(1);
  const scrollAnchorRef = useRef<{ page: number; offset: number } | null>(null);

  // Load PDF and measure all pages
  useEffect(() => {
    let cancelled = false;
    const loadingTask = pdfjsLib.getDocument(documentPdfUrl(docId));

    loadingTask.promise.then(async (pdf) => {
      if (cancelled) return;
      const numPages = pdf.numPages;
      const container = containerRef.current;
      if (!container) return;

      const availableWidth = Math.max(280, container.clientWidth - 24);
      const heights: number[] = [];

      for (let i = 1; i <= numPages; i++) {
        const page = await pdf.getPage(i);
        if (cancelled) return;
        const viewport = page.getViewport({ scale: 1 });
        const scale = Math.min(1.7, Math.max(0.55, availableWidth / viewport.width));
        scaleRef.current = scale;
        const scaledViewport = page.getViewport({ scale });
        heights.push(scaledViewport.height);
      }

      if (!cancelled) {
        setTotalPages(numPages);
        setPageHeights(heights);
      }
    }).catch(() => {});

    return () => {
      cancelled = true;
      void loadingTask.destroy();
    };
  }, [docId]);

  // Compute total height
  const totalHeight = useMemo(() => {
    if (!pageHeights.length) return 0;
    return pageHeights.reduce((sum, h) => sum + h, 0) + (pageHeights.length - 1) * PAGE_GAP;
  }, [pageHeights]);

  // Get Y offset for a given page (1-based)
  const getPageOffset = useCallback(
    (page: number) => {
      if (!pageHeights.length) return 0;
      let offset = 0;
      for (let i = 0; i < page - 1 && i < pageHeights.length; i++) {
        offset += pageHeights[i] + PAGE_GAP;
      }
      return offset;
    },
    [pageHeights],
  );

  // Track visible page on scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !pageHeights.length) return;

    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const viewCenter = scrollTop + container.clientHeight / 2;
      let offset = 0;
      for (let i = 0; i < pageHeights.length; i++) {
        const pageBottom = offset + pageHeights[i];
        if (viewCenter >= offset && viewCenter <= pageBottom) {
          onVisiblePageChange(i + 1);
          break;
        }
        offset = pageHeights[i] + PAGE_GAP + offset;
      }
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => container.removeEventListener("scroll", handleScroll);
  }, [pageHeights, onVisiblePageChange]);

  // Compute which pages to render based on scroll position
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !pageHeights.length) return;

    const updateRange = () => {
      const scrollTop = container.scrollTop;
      const scrollBottom = scrollTop + container.clientHeight;

      let offset = 0;
      let start = 0;
      let end = 0;
      let foundStart = false;

      for (let i = 0; i < pageHeights.length; i++) {
        const pageTop = offset;
        const pageBottom = offset + pageHeights[i];

        if (!foundStart && pageBottom >= scrollTop) {
          start = Math.max(0, i - BUFFER_PAGES);
          foundStart = true;
        }
        if (foundStart && pageTop > scrollBottom) {
          end = Math.min(pageHeights.length, i + BUFFER_PAGES);
          break;
        }
        offset = pageBottom + PAGE_GAP;
      }
      if (end === 0) end = pageHeights.length;
      setRenderRange({ start, end });
    };

    updateRange();
    container.addEventListener("scroll", updateRange, { passive: true });
    return () => container.removeEventListener("scroll", updateRange);
  }, [pageHeights]);

  // Scroll to specific page when scrollToPage changes
  useEffect(() => {
    if (!scrollToPage || !pageHeights.length || !containerRef.current) return;
    const container = containerRef.current;
    const targetPage = Math.min(Math.max(1, scrollToPage), pageHeights.length);
    const offset = getPageOffset(targetPage);
    container.scrollTo({ top: offset, behavior: "smooth" });
  }, [scrollToPage, pageHeights, getPageOffset]);

  return (
    <div className="pdf-scroll-container" ref={containerRef}>
      <div className="pdf-scroll-spacer" style={{ height: totalHeight, position: "relative" }}>
        {pageHeights.length > 0 &&
          Array.from({ length: renderRange.end - renderRange.start }, (_, i) => {
            const pageIndex = renderRange.start + i;
            const pageNum = pageIndex + 1;
            let top = 0;
            for (let j = 0; j < pageIndex; j++) {
              top += pageHeights[j] + PAGE_GAP;
            }
            return (
              <div
                key={pageNum}
                className="pdf-page-slot"
                style={{ position: "absolute", top, left: 0, right: 0 }}
              >
                <PdfPageCanvas docId={docId} page={pageNum} scale={scaleRef.current} />
                <span className="pdf-page-number">{pageNum}</span>
              </div>
            );
          })}
      </div>
    </div>
  );
}

function PdfPageCanvas({ docId, page, scale }: { docId: string; page: number; scale: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let loadingTask: pdfjsLib.PDFDocumentLoadingTask | null = null;
    let renderTask: { cancel: () => void; promise: Promise<unknown> } | null = null;

    async function renderPage() {
      const canvas = canvasRef.current;
      if (!canvas || !docId) return;

      setError("");
      loadingTask = pdfjsLib.getDocument(documentPdfUrl(docId));
      const pdf = await loadingTask.promise;
      if (cancelled) return;
      const boundedPage = Math.min(Math.max(1, page), pdf.numPages);
      const pdfPage = await pdf.getPage(boundedPage);
      if (cancelled) return;

      const viewport = pdfPage.getViewport({ scale });
      const pixelRatio = window.devicePixelRatio || 1;
      const context = canvas.getContext("2d");
      if (!context) return;

      canvas.width = Math.floor(viewport.width * pixelRatio);
      canvas.height = Math.floor(viewport.height * pixelRatio);
      canvas.style.width = `${Math.floor(viewport.width)}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, viewport.width, viewport.height);

      renderTask = pdfPage.render({ canvasContext: context, viewport });
      await renderTask.promise;
    }

    renderPage().catch((err: unknown) => {
      if (cancelled) return;
      setError(err instanceof Error ? err.message : "PDF 渲染失败");
    });

    return () => {
      cancelled = true;
      renderTask?.cancel();
      void loadingTask?.destroy();
    };
  }, [docId, page, scale]);

  return (
    <div className="pdf-canvas-wrap">
      <canvas ref={canvasRef} />
      {error ? <p className="pdf-error">{error}</p> : null}
    </div>
  );
}

function DocumentPanel({
  documents,
  activeDocId,
  scrollToPage,
  pageDetail,
  collapsed,
  onToggle,
  onOpenPage,
  onVisiblePageChange,
  onResize,
  onResizeStart,
  onResizeEnd,
}: {
  documents: DocumentResponse[];
  activeDocId: string;
  scrollToPage: number;
  pageDetail: DocumentPageResponse | null;
  collapsed: boolean;
  onToggle: () => void;
  onOpenPage: (docId: string, page: number) => void;
  onVisiblePageChange: (page: number) => void;
  onResize: (width: number) => void;
  onResizeStart: () => void;
  onResizeEnd: () => void;
}) {
  const handleRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handle = handleRef.current;
    if (!handle || collapsed) return;

    let startX = 0;
    let startWidth = 0;

    const onMouseMove = (e: MouseEvent) => {
      const delta = startX - e.clientX;
      const next = Math.min(900, Math.max(280, startWidth + delta));
      onResize(next);
    };

    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      onResizeEnd();
    };

    const onMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      startX = e.clientX;
      startWidth = handle.parentElement?.clientWidth ?? 420;
      onResizeStart();
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    };

    handle.addEventListener("mousedown", onMouseDown);
    return () => {
      handle.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
  }, [collapsed, onResize, onResizeStart, onResizeEnd]);
  const activeDoc = documents.find((doc) => doc.id === activeDocId) ?? documents[0] ?? null;

  if (collapsed) {
    return (
      <aside className="document-panel collapsed" aria-label="PDF 预览">
        <button type="button" className="document-panel-toggle" onClick={onToggle} aria-label="展开 PDF 预览">
          <Icon name="panel" />
        </button>
      </aside>
    );
  }

  return (
    <aside className="document-panel" aria-label="PDF 预览">
      <div className="resize-handle" ref={handleRef} />
      <div className="document-panel-header">
        <div>
          <span>Source</span>
          <h2>PDF 预览</h2>
        </div>
        <button type="button" onClick={onToggle} aria-label="收起 PDF 预览">
          <Icon name="panel" />
        </button>
      </div>

      <div className="document-controls">
        <label>
          <span>报告</span>
          <select
            value={activeDoc?.id ?? ""}
            onChange={(event) => onOpenPage(event.target.value, 1)}
          >
            {documents.length === 0 ? <option value="">暂无报告</option> : null}
            {documents.map((doc) => (
              <option value={doc.id} key={doc.id}>
                {doc.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="pdf-frame-wrap">
        {activeDoc ? (
          <PdfScrollViewer
            docId={activeDoc.id}
            scrollToPage={scrollToPage}
            onVisiblePageChange={onVisiblePageChange}
            key={activeDoc.id}
          />
        ) : (
          <div className="pdf-empty">
            <Icon name="file" />
            <p>没有找到可预览的 PDF</p>
          </div>
        )}
      </div>

      <div className="page-excerpt">
        <div className="page-excerpt-title">
          <strong>{activeDoc?.name ?? "未选择报告"}</strong>
          {activeDoc ? <span>p.{pageDetail?.page ?? "-"} / {activeDoc.page_count}</span> : null}
        </div>
        <p>{pageDetail?.text || "滚动 PDF 查看内容，或点击引用跳转到指定页面。"}</p>
      </div>
    </aside>
  );
}

function ToolResults({ results }: { results: ToolResultResponse[] }) {
  if (!results.length) return null;
  return (
    <section className="tool-results" aria-label="工具调用">
      {results.map((result) => (
        <div className={`tool-result ${result.status}`} key={result.id}>
          <span>{result.name}</span>
          <strong>{result.status === "running" ? "运行中" : result.status === "error" ? "失败" : "完成"}</strong>
          {result.error ? <em>{result.error}</em> : null}
        </div>
      ))}
    </section>
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
    listSessions().then(setSessions).catch(() => {});
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
      .catch(() => {});
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
          .catch(() => setPageDetail(null));
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
        await streamChat({ question, session_id: sessionId }, (event) => {
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
    [input, isSending, sessionId, updateAssistant, refreshSessions],
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
