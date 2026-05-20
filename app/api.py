import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chat_service import ChatService
from app.factory import build_chat_service_from_env
from app.session import SQLiteSessionStore, SessionSummary, SessionTurn


class UsageResponse(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    audio_tokens: int = 0
    image_tokens: int = 0
    video_tokens: int = 0
    context_window_tokens: int = 0
    context_used_tokens: int = 0
    context_ratio: float = 0.0
    estimated: bool = False


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


class CitationResponse(BaseModel):
    doc_id: str
    doc_name: str
    page: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_content: str = ""
    usage: UsageResponse | None = None


class SessionSummaryResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class SessionMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_content: str = ""
    usage: UsageResponse | None = None
    created_at: str


class SessionDetailResponse(BaseModel):
    session: SessionSummaryResponse
    messages: list[SessionMessageResponse] = Field(default_factory=list)


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)
    if not isinstance(service, ChatService):
        raise HTTPException(status_code=503, detail="Chat service is not initialized")
    return service


def get_session_store(request: Request) -> SQLiteSessionStore:
    store = getattr(request.app.state, "session_store", None)
    if not isinstance(store, SQLiteSessionStore):
        raise HTTPException(status_code=503, detail="Session store is not initialized")
    return store


def _session_summary_response(session: SessionSummary) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _citation_response(item: dict[str, Any]) -> CitationResponse:
    return CitationResponse(
        doc_id=str(item.get("doc_id", "")),
        doc_name=str(item.get("doc_name", "")),
        page=item.get("page") if isinstance(item.get("page"), int) else None,
    )


def _messages_from_turns(turns: list[SessionTurn]) -> list[SessionMessageResponse]:
    messages: list[SessionMessageResponse] = []
    for turn in turns:
        messages.append(
            SessionMessageResponse(
                id=f"{turn.id}:user",
                role="user",
                content=turn.user_content,
                created_at=turn.created_at,
            )
        )
        messages.append(
            SessionMessageResponse(
                id=f"{turn.id}:assistant",
                role="assistant",
                content=turn.assistant_content,
                citations=[_citation_response(item) for item in turn.citations],
                tool_results=turn.tool_results,
                reasoning_content=turn.reasoning_content,
                usage=UsageResponse(**turn.usage) if isinstance(turn.usage, dict) else None,
                created_at=turn.created_at,
            )
        )
    return messages


def _chat_response(payload: dict[str, Any]) -> ChatResponse:
    usage = payload.get("usage")
    return ChatResponse(
        session_id=str(payload.get("session_id", "default")),
        answer=str(payload.get("answer", "")),
        citations=[_citation_response(item) for item in payload.get("citations", [])],
        tool_results=[item for item in payload.get("tool_results", []) if isinstance(item, dict)],
        reasoning_content=str(payload.get("reasoning_content", "")),
        usage=UsageResponse(**usage) if isinstance(usage, dict) else None,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    chat_service: ChatService | None = None,
    session_store: SQLiteSessionStore | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_chat_service = chat_service is None
        resolved_session_store = session_store or SQLiteSessionStore.from_env()
        resolved_chat_service = chat_service or build_chat_service_from_env(
            session_store=resolved_session_store
        )
        app.state.session_store = resolved_session_store
        app.state.chat_service = resolved_chat_service
        try:
            yield
        finally:
            if owns_chat_service:
                resolved_chat_service.close()

    app = FastAPI(
        title="Fintell Chat API",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sessions", response_model=list[SessionSummaryResponse])
    def list_sessions(
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> list[SessionSummaryResponse]:
        return [_session_summary_response(s) for s in store.list_sessions()]

    @app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
    def get_session(
        session_id: str,
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> SessionDetailResponse:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionDetailResponse(
            session=_session_summary_response(session),
            messages=_messages_from_turns(store.list_turns(session_id)),
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        service: ChatService = Depends(get_chat_service),
    ) -> ChatResponse:
        try:
            result = service.ask(
                payload.question,
                session_id=payload.session_id or "default",
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _chat_response(result.to_dict())

    @app.post("/chat/stream")
    def chat_stream(
        payload: ChatRequest,
        service: ChatService = Depends(get_chat_service),
    ) -> StreamingResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")

        def event_stream():
            try:
                for item in service.stream(
                    question,
                    session_id=payload.session_id or "default",
                ):
                    yield _sse(item.get("event", "status"), item.get("data", {}))
            except Exception as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()
