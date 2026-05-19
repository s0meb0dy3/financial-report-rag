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
};

export type ChatResponse = {
  session_id: string;
  answer: string;
  citations: CitationResponse[];
  reasoning_content: string;
  usage: UsageResponse | null;
};

export type SessionMessageResponse = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: CitationResponse[];
  reasoning_content: string;
  tool_results: ToolResultResponse[];
  usage: UsageResponse | null;
  created_at: string;
};

export type ToolResultResponse = {
  id?: string;
  name?: string;
  status?: "running" | "done" | "error";
  message?: string;
  evidence_count?: number;
  table_count?: number;
  citation_count?: number;
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

export type ChatStreamEvent =
  | { event: "session"; data: { session_id: string } }
  | { event: "status"; data: { message: string } }
  | {
      event: "tool";
      data: {
        id: string;
        name: string;
        status: "running" | "done" | "error";
        message: string;
        evidence_count?: number;
        table_count?: number;
        citation_count?: number;
      };
    }
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
    eventName === "tool" ||
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
