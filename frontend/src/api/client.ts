export type CitationResponse = {
  doc_id: string;
  doc_name: string;
  page: number | null;
};

export type UsageResponse = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  reasoning_tokens: number;
  cached_tokens: number;
  audio_tokens: number;
  image_tokens: number;
  video_tokens: number;
  context_window_tokens: number;
  context_used_tokens: number;
  context_ratio: number;
  estimated: boolean;
};

export type ChatRequest = {
  question: string;
  session_id?: string | null;
  doc_id?: string | null;
  visible_page?: number | null;
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  citations: CitationResponse[];
  tool_results: ToolResultResponse[];
  reasoning_content: string;
  usage: UsageResponse | null;
};

export type ToolResultResponse = {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "running" | "done" | "error";
  content?: Record<string, unknown>;
  citations?: CitationResponse[];
  error?: string;
};

export type SessionMessageResponse = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: CitationResponse[];
  tool_results: ToolResultResponse[];
  reasoning_content: string;
  usage: UsageResponse | null;
  created_at: string;
};

export type SessionDetailResponse = {
  session: {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
  };
  messages: SessionMessageResponse[];
};

export type DocumentResponse = {
  id: string;
  name: string;
  page_count: number;
  parsed: boolean;
};

export type DocumentPageResponse = {
  doc_id: string;
  doc_name: string;
  page: number;
  text: string;
  blocks: Array<{
    type: string;
    text: string;
    bbox: Array<number> | null;
  }>;
};

export type ChatStreamEvent =
  | { event: "session"; data: { session_id: string } }
  | { event: "status"; data: { message: string } }
  | { event: "tool_call"; data: ToolResultResponse }
  | { event: "tool_result"; data: ToolResultResponse }
  | { event: "reasoning_delta"; data: { content: string } }
  | { event: "answer_delta"; data: { content: string } }
  | { event: "usage"; data: UsageResponse }
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

export type SessionSummaryResponse = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export function listSessions() {
  return requestJson<SessionSummaryResponse[]>("/sessions");
}

export function renameSession(sessionId: string, title: string) {
  return requestJson<SessionSummaryResponse>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
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

export function listDocuments() {
  return requestJson<DocumentResponse[]>("/documents");
}

export function getDocumentPage(docId: string, page: number) {
  return requestJson<DocumentPageResponse>(
    `/documents/${encodeURIComponent(docId)}/pages/${page}`,
  );
}

export function uploadDocument(file: File) {
  return requestJson<DocumentResponse>(`/documents?filename=${encodeURIComponent(file.name)}`, {
    method: "POST",
    headers: { "Content-Type": "application/pdf" },
    body: file,
  });
}

export async function deleteDocument(docId: string) {
  const response = await fetch(`${API_PREFIX}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw await responseError(response);
  }
}

export function documentPdfUrl(docId: string, page?: number | null) {
  const base = `${API_PREFIX}/documents/${encodeURIComponent(docId)}/pdf`;
  return page ? `${base}#page=${page}` : base;
}

export function getSession(sessionId: string) {
  return requestJson<SessionDetailResponse>(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function chat(payload: ChatRequest) {
  return requestJson<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function streamChat(
  payload: ChatRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal,
) {
  const response = await fetch(`${API_PREFIX}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
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
    eventName === "tool_call" ||
    eventName === "tool_result" ||
    eventName === "reasoning_delta" ||
    eventName === "answer_delta" ||
    eventName === "usage" ||
    eventName === "final" ||
    eventName === "error"
  ) {
    return { event: eventName, data } as ChatStreamEvent;
  }
  return null;
}
