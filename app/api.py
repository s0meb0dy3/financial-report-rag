import json
from contextlib import asynccontextmanager
from typing import Any, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import AgentLoop
from app.session import SQLiteSessionStore, SessionSummary, SessionTurn


class CitationResponse(BaseModel):
    doc_id: str
    doc_name: str
    page: int | None = None


class ToolTraceResponse(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    tool_call_id: str = ""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_id: str | None = None
    include_tool_results: bool = False


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    tool_results: list[ToolTraceResponse] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    doc_id: str
    doc_name: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class SessionSummaryResponse(BaseModel):
    id: str
    title: str
    doc_id: str | None = None
    created_at: str
    updated_at: str


class SessionMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    tool_results: list[ToolTraceResponse] = Field(default_factory=list)
    created_at: str


class SessionDetailResponse(BaseModel):
    session: SessionSummaryResponse
    messages: list[SessionMessageResponse] = Field(default_factory=list)


class SessionsResponse(BaseModel):
    sessions: list[SessionSummaryResponse]


class CreateSessionRequest(BaseModel):
    title: str | None = None
    doc_id: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    doc_id: str | None = None


def get_agent_loop(request: Request) -> AgentLoop:
    loop = getattr(request.app.state, "agent_loop", None)
    if loop is None:
        raise HTTPException(status_code=503, detail="Agent loop is not initialized")
    return loop


def get_session_store(request: Request) -> SQLiteSessionStore:
    store = getattr(request.app.state, "session_store", None)
    if not isinstance(store, SQLiteSessionStore):
        raise HTTPException(status_code=503, detail="Session store is not initialized")
    return store


def _session_summary_response(session: SessionSummary) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        id=session.id,
        title=session.title,
        doc_id=session.doc_id,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _tool_trace_response(item: dict[str, Any]) -> ToolTraceResponse:
    return ToolTraceResponse(
        tool_name=item.get("tool_name", ""),
        arguments=item.get("arguments", {}),
        output=item.get("output", {}),
        tool_call_id=item.get("tool_call_id", ""),
    )


def _citation_response(item: dict[str, Any]) -> CitationResponse:
    return CitationResponse(
        doc_id=item.get("doc_id", ""),
        doc_name=item.get("doc_name", ""),
        page=item.get("page"),
    )


def _chat_response_from_result(
    result: dict[str, Any],
    *,
    include_tool_results: bool,
) -> ChatResponse:
    tool_results = result.get("tool_results", []) if include_tool_results else []
    return ChatResponse(
        answer=result.get("answer", ""),
        citations=[_citation_response(item) for item in result.get("citations", [])],
        tool_results=[_tool_trace_response(item) for item in tool_results],
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
                tool_results=[_tool_trace_response(item) for item in turn.tool_results],
                created_at=turn.created_at,
            )
        )
    return messages


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    agent_loop: Optional[AgentLoop] = None,
    session_store: SQLiteSessionStore | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_loop = agent_loop is None
        resolved_session_store = session_store
        if agent_loop is None and resolved_session_store is None:
            resolved_session_store = SQLiteSessionStore.from_env()
        app.state.session_store = resolved_session_store
        app.state.agent_loop = agent_loop or AgentLoop.from_env(
            session_store=resolved_session_store
        )
        try:
            yield
        finally:
            if owns_loop:
                app.state.agent_loop.close()

    app = FastAPI(
        title="Financial Report Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/documents", response_model=DocumentsResponse)
    def list_documents(loop: AgentLoop = Depends(get_agent_loop)) -> DocumentsResponse:
        documents = loop.retriever.list_documents()
        return DocumentsResponse(
            documents=[
                DocumentResponse(doc_id=document.doc_id, doc_name=document.doc_name)
                for document in documents
            ]
        )

    @app.get("/sessions", response_model=SessionsResponse)
    def list_sessions(store: SQLiteSessionStore = Depends(get_session_store)) -> SessionsResponse:
        return SessionsResponse(
            sessions=[_session_summary_response(item) for item in store.list_sessions()]
        )

    @app.post("/sessions", response_model=SessionSummaryResponse)
    def create_session(
        payload: CreateSessionRequest,
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> SessionSummaryResponse:
        session = store.create_session(
            str(uuid4()),
            title=payload.title or "新的财报对话",
            doc_id=payload.doc_id,
        )
        return _session_summary_response(session)

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

    @app.patch("/sessions/{session_id}", response_model=SessionSummaryResponse)
    def update_session(
        session_id: str,
        payload: UpdateSessionRequest,
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> SessionSummaryResponse:
        if store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session = store.update_session(
            session_id,
            title=payload.title,
            doc_id=payload.doc_id,
        )
        return _session_summary_response(session)

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(
        session_id: str,
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> Response:
        if not store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return Response(status_code=204)

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, loop: AgentLoop = Depends(get_agent_loop)) -> ChatResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")

        result = loop.run_turn(
            question,
            session_id=payload.session_id,
            top_k=payload.top_k,
            doc_id=payload.doc_id,
        )
        return _chat_response_from_result(
            result,
            include_tool_results=payload.include_tool_results,
        )

    @app.post("/chat/stream")
    def chat_stream(
        payload: ChatRequest,
        loop: AgentLoop = Depends(get_agent_loop),
    ) -> StreamingResponse:
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")

        def event_stream():
            try:
                for item in loop.run_turn_stream(
                    question,
                    session_id=payload.session_id,
                    top_k=payload.top_k,
                    doc_id=payload.doc_id,
                ):
                    event = item.get("event", "status")
                    data = item.get("data", {})
                    if event == "final":
                        response = _chat_response_from_result(
                            data,
                            include_tool_results=True,
                        )
                        data = response.model_dump()
                    yield _sse(event, data)
            except Exception as exc:
                yield _sse("error", {"message": str(exc)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()
