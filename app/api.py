import json
import logging
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.chat_service import ChatService
from app.config import AppConfig
from app.documents import DocumentService, DocumentServiceError
from app.factory import build_chat_service_from_env
from app.session import SQLiteSessionStore, SessionSummary, SessionTurn


logger = logging.getLogger(__name__)


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
    doc_id: str | None = None
    visible_page: int | None = None


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


class SessionRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class DocumentResponse(BaseModel):
    id: str
    name: str
    page_count: int
    parsed: bool = True


class DocumentPageBlockResponse(BaseModel):
    type: str
    text: str = ""
    bbox: list[float | int] | None = None


class DocumentTocEntryResponse(BaseModel):
    level: int
    title: str
    page: int
    page_label: str | None = None


class DocumentTocResponse(BaseModel):
    doc_id: str
    doc_name: str
    page_count: int
    summary: str
    entries: list[DocumentTocEntryResponse] = Field(default_factory=list)


class DocumentPageResponse(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    text: str
    blocks: list[DocumentPageBlockResponse] = Field(default_factory=list)


class RuntimeConfigResponse(BaseModel):
    status: str = "ok"
    chat_model: str
    chat_base_url: str
    api_key_configured: bool
    mineru_api_key_configured: bool


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


def get_document_service(request: Request) -> DocumentService:
    service = getattr(request.app.state, "document_service", None)
    if not isinstance(service, DocumentService):
        raise HTTPException(status_code=503, detail="Document service is not initialized")
    return service


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
    document_service: DocumentService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_chat_service = chat_service is None
        config = AppConfig.from_env()
        resolved_session_store = session_store or SQLiteSessionStore.from_env()
        resolved_document_service = document_service or DocumentService(mineru_api_key=config.mineru_api_key)
        resolved_chat_service = chat_service or build_chat_service_from_env(
            session_store=resolved_session_store,
            document_service=resolved_document_service,
        )
        app.state.session_store = resolved_session_store
        app.state.document_service = resolved_document_service
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

    @app.get("/runtime/config", response_model=RuntimeConfigResponse)
    def runtime_config() -> RuntimeConfigResponse:
        config = AppConfig.from_env()
        return RuntimeConfigResponse(
            chat_model=config.chat_model,
            chat_base_url=config.chat_base_url,
            api_key_configured=bool(config.chat_api_key),
            mineru_api_key_configured=bool(config.mineru_api_key),
        )

    @app.get("/sessions", response_model=list[SessionSummaryResponse])
    def list_sessions(
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> list[SessionSummaryResponse]:
        return [_session_summary_response(s) for s in store.list_sessions()]

    @app.get("/documents", response_model=list[DocumentResponse])
    def list_documents(
        service: DocumentService = Depends(get_document_service),
    ) -> list[DocumentResponse]:
        return [DocumentResponse(**doc.to_dict()) for doc in service.list_documents()]

    @app.get("/documents/{doc_id}/toc", response_model=DocumentTocResponse)
    def get_document_toc(
        doc_id: str,
        service: DocumentService = Depends(get_document_service),
    ) -> DocumentTocResponse:
        try:
            return DocumentTocResponse(**service.get_toc(doc_id))
        except DocumentServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/documents/{doc_id}/pdf")
    def get_document_pdf(
        doc_id: str,
        service: DocumentService = Depends(get_document_service),
    ) -> FileResponse:
        try:
            doc = service.get_document(doc_id)
            pdf_path = service.get_pdf_path(doc_id)
        except DocumentServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(doc.name)}"},
        )

    @app.get("/documents/{doc_id}/pages/{page}", response_model=DocumentPageResponse)
    def get_document_page(
        doc_id: str,
        page: int,
        service: DocumentService = Depends(get_document_service),
    ) -> DocumentPageResponse:
        try:
            parsed_page = service.read_page(doc_id, page)
        except DocumentServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return DocumentPageResponse(**parsed_page.to_dict())

    @app.post("/documents", response_model=DocumentResponse)
    async def upload_document(
        request: Request,
        filename: str,
        service: DocumentService = Depends(get_document_service),
    ) -> DocumentResponse:
        content_type = request.headers.get("content-type", "")
        if content_type and "application/pdf" not in content_type.lower():
            raise HTTPException(status_code=415, detail="Only application/pdf uploads are supported")
        try:
            doc = service.save_upload(filename, await request.body())
        except DocumentServiceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return DocumentResponse(**doc.to_dict())

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(
        doc_id: str,
        service: DocumentService = Depends(get_document_service),
    ) -> Response:
        try:
            removed = service.delete_document(doc_id)
        except DocumentServiceError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=403, detail="Document is outside managed storage")
        return Response(status_code=204)

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
    def rename_session(
        session_id: str,
        payload: SessionRenameRequest,
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> SessionSummaryResponse:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _session_summary_response(store.update_session(session_id, title=payload.title))

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(
        session_id: str,
        store: SQLiteSessionStore = Depends(get_session_store),
    ) -> Response:
        if not store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return Response(status_code=204)

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        service: ChatService = Depends(get_chat_service),
    ) -> ChatResponse:
        try:
            result = service.ask(
                payload.question,
                session_id=payload.session_id or "default",
                doc_id=payload.doc_id,
                visible_page=payload.visible_page,
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
                    doc_id=payload.doc_id,
                    visible_page=payload.visible_page,
                ):
                    yield _sse(item.get("event", "status"), item.get("data", {}))
            except Exception as exc:
                logger.error("chat stream failed: %s", type(exc).__name__)
                yield _sse("error", {"message": "请求失败，请查看后端日志。"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()
