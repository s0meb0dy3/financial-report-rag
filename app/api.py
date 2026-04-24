from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent import AgentLoop


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


def get_agent_loop(request: Request) -> AgentLoop:
    loop = getattr(request.app.state, "agent_loop", None)
    if loop is None:
        raise HTTPException(status_code=503, detail="Agent loop is not initialized")
    return loop


def create_app(agent_loop: Optional[AgentLoop] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owns_loop = agent_loop is None
        app.state.agent_loop = agent_loop or AgentLoop.from_env()
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
        tool_results = result.get("tool_results", []) if payload.include_tool_results else []
        return ChatResponse(
            answer=result.get("answer", ""),
            citations=[
                CitationResponse(
                    doc_id=item.get("doc_id", ""),
                    doc_name=item.get("doc_name", ""),
                    page=item.get("page"),
                )
                for item in result.get("citations", [])
            ],
            tool_results=[
                ToolTraceResponse(
                    tool_name=item.get("tool_name", ""),
                    arguments=item.get("arguments", {}),
                    output=item.get("output", {}),
                    tool_call_id=item.get("tool_call_id", ""),
                )
                for item in tool_results
            ],
        )

    return app


app = create_app()
