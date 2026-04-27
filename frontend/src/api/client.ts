import type { components } from "./schema";

export type ChatRequest = components["schemas"]["ChatRequest"];
export type ChatResponse = components["schemas"]["ChatResponse"];
export type CitationResponse = components["schemas"]["CitationResponse"];
export type CreateSessionRequest = components["schemas"]["CreateSessionRequest"];
export type DocumentJobResponse = components["schemas"]["DocumentJobResponse"];
export type DocumentJobsResponse = components["schemas"]["DocumentJobsResponse"];
export type DocumentResponse = components["schemas"]["DocumentResponse"];
export type DocumentsResponse = components["schemas"]["DocumentsResponse"];
export type SessionDetailResponse = components["schemas"]["SessionDetailResponse"];
export type SessionMessageResponse = components["schemas"]["SessionMessageResponse"];
export type SessionSummaryResponse = components["schemas"]["SessionSummaryResponse"];
export type SessionsResponse = components["schemas"]["SessionsResponse"];
export type ToolTraceResponse = components["schemas"]["ToolTraceResponse"];
export type UpdateSessionRequest = components["schemas"]["UpdateSessionRequest"];

export type ChatStreamEvent =
  | { event: "session"; data: { session_id: string } }
  | { event: "status"; data: { message: string } }
  | { event: "tool_result"; data: ToolTraceResponse }
  | { event: "answer_delta"; data: { content: string } }
  | { event: "final"; data: ChatResponse }
  | { event: "error"; data: { message: string } };

const API_PREFIX = "/api";

async function responseError(response: Response) {
  const text = await response.text();
  if (!text) {
    return new Error(`${response.status} ${response.statusText}`);
  }
  try {
    const payload = JSON.parse(text) as { detail?: unknown; message?: unknown };
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload.message === "string"
          ? payload.message
          : text;
    return new Error(detail);
  } catch {
    return new Error(text);
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  return (await response.json()) as T;
}

export function listDocuments() {
  return requestJson<DocumentsResponse>("/documents");
}

export async function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_PREFIX}/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  return (await response.json()) as DocumentJobResponse;
}

export function listDocumentJobs() {
  return requestJson<DocumentJobsResponse>("/documents/jobs");
}

export function getDocumentJob(jobId: string) {
  return requestJson<DocumentJobResponse>(`/documents/jobs/${encodeURIComponent(jobId)}`);
}

export async function deleteDocument(docId: string) {
  const response = await fetch(`${API_PREFIX}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw await responseError(response);
  }
}

export function listSessions() {
  return requestJson<SessionsResponse>("/sessions");
}

export function createSession(payload: CreateSessionRequest) {
  return requestJson<SessionSummaryResponse>("/sessions", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSession(sessionId: string) {
  return requestJson<SessionDetailResponse>(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function updateSession(
  sessionId: string,
  payload: UpdateSessionRequest,
) {
  return requestJson<SessionSummaryResponse>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function deleteSession(sessionId: string) {
  const response = await fetch(`${API_PREFIX}/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw await responseError(response);
  }
}

export async function streamChat(
  payload: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
) {
  const response = await fetch(`${API_PREFIX}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  if (!response.body) {
    throw new Error("浏览器没有收到流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const event = parseSseBlock(block);
      if (!event) continue;
      onEvent(event);
      if (event.event === "error") {
        throw new Error(event.data.message || "流式请求失败。");
      }
    }
  }

  buffer += decoder.decode();
  const trailing = parseSseBlock(buffer.trim());
  if (trailing) {
    onEvent(trailing);
    if (trailing.event === "error") {
      throw new Error(trailing.data.message || "流式请求失败。");
    }
  }
}

function parseSseBlock(block: string): ChatStreamEvent | null {
  if (!block.trim()) return null;

  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (!dataLines.length) return null;
  const data = JSON.parse(dataLines.join("\n")) as unknown;
  if (
    eventName === "session" ||
    eventName === "status" ||
    eventName === "tool_result" ||
    eventName === "answer_delta" ||
    eventName === "final" ||
    eventName === "error"
  ) {
    return { event: eventName, data } as ChatStreamEvent;
  }
  return null;
}
