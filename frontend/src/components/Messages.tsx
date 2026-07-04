import { useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

import type { CitationResponse, ToolResultResponse, UsageResponse } from "../api/client";
import { EChart } from "./EChart";
import { Icon } from "./Icon";

export type Message = {
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
    .replace(/\$\$\s*([^\n]+?)\s*\$\$/g, (_, formula: string) => `\n\n$$\n${formula.trim()}\n$$\n\n`)
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
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
      {normalizeMarkdown(content)}
    </ReactMarkdown>
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

export function UsageBar({ usage, status }: { usage: UsageResponse | null; status: string }) {
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

export function AssistantMessage({
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

function ToolResults({ results }: { results: ToolResultResponse[] }) {
  if (!results.length) return null;
  return (
    <section className="tool-results" aria-label="工具调用">
      {results.map((result) => {
        const chartOption = result.content?.chart_option as import("echarts").EChartsOption | undefined;
        const isChart =
          result.name === "create_chart" &&
          result.status === "done" &&
          chartOption;
        return (
          <div className={`tool-result ${result.status}`} key={result.id}>
            {isChart ? (
              <EChart option={chartOption} />
            ) : (
              <ToolResultCard result={result} />
            )}
          </div>
        );
      })}
    </section>
  );
}

function ToolResultCard({ result }: { result: ToolResultResponse }) {
  const [open, setOpen] = useState(false);
  const title = toolTitle(result.name);
  const summary = toolSummary(result);
  const results = Array.isArray(result.content?.results) ? result.content.results : [];

  return (
    <>
      <button className="tool-result-header" type="button" onClick={() => setOpen((v) => !v)}>
        <span>{title}</span>
        <strong>{result.status === "running" ? "运行中" : result.status === "error" ? "失败" : "完成"}</strong>
        <Icon name="chevron" className={open ? "chevron open" : "chevron"} />
      </button>
      {summary ? <p className="tool-result-summary">{summary}</p> : null}
      {result.error ? <em>{result.error}</em> : null}
      {results.length ? (
        <div className="tool-search-hits">
          {results.slice(0, 3).map((item, index) => renderSearchHit(item, index))}
        </div>
      ) : null}
      {open ? (
        <pre className="tool-result-json">{JSON.stringify({ arguments: result.arguments, content: result.content }, null, 2)}</pre>
      ) : null}
    </>
  );
}

function toolTitle(name: string) {
  const titles: Record<string, string> = {
    list_reports: "列出报告",
    read_toc: "读取目录",
    read_pdf_page: "读取页面",
    search_report_text: "搜索报告",
    tavily_search: "联网搜索",
  };
  return titles[name] ?? name;
}

function toolSummary(result: ToolResultResponse) {
  const args = result.arguments ?? {};
  if (result.name === "search_report_text") {
    const count = Array.isArray(result.content?.results) ? result.content.results.length : 0;
    return `关键词：${String(args.query ?? result.content?.query ?? "-")}，命中 ${count} 条`;
  }
  if (result.name === "read_pdf_page") {
    return `文档：${String(args.doc_id ?? result.content?.doc_name ?? "-")}，页码：${String(args.page ?? result.content?.page ?? "-")}`;
  }
  if (result.name === "read_toc") return `文档：${String(args.doc_id ?? result.content?.doc_name ?? "-")}`;
  if (result.name === "list_reports") {
    const count = Array.isArray(result.content?.reports) ? result.content.reports.length : 0;
    return `可用报告 ${count} 份`;
  }
  return Object.keys(args).length ? JSON.stringify(args) : "";
}

function renderSearchHit(item: unknown, index: number) {
  if (!item || typeof item !== "object") return null;
  const hit = item as Record<string, unknown>;
  const docName = String(hit.doc_name ?? hit.doc_id ?? "报告");
  const page = hit.page ? ` p.${String(hit.page)}` : "";
  const snippet = String(hit.snippet ?? "");
  const terms = Array.isArray(hit.matched_terms) ? hit.matched_terms.map(String) : [];
  return (
    <div className="tool-search-hit" key={`${docName}-${page}-${index}`}>
      <b>{docName}{page}</b>
      <span><HighlightedSnippet text={snippet} terms={terms} /></span>
    </div>
  );
}

function HighlightedSnippet({ text, terms }: { text: string; terms: string[] }) {
  const usefulTerms = terms.filter(Boolean).sort((a, b) => b.length - a.length);
  if (!usefulTerms.length || !text) return <>{text}</>;
  const pattern = new RegExp(`(${usefulTerms.map(escapeRegExp).join("|")})`, "gi");
  return (
    <>
      {text.split(pattern).map((part, index) =>
        usefulTerms.some((term) => term.toLowerCase() === part.toLowerCase()) ? <mark key={index}>{part}</mark> : part,
      )}
    </>
  );
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function UserMessage({ content }: { content: string }) {
  return (
    <article className="user-message">
      <div className="user-bubble">{content}</div>
      <div className="user-actions" aria-label="用户消息操作">
        <CopyButton text={content} label="复制问题" />
      </div>
    </article>
  );
}
