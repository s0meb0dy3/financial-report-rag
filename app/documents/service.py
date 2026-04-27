import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from app.domain import DocumentRef
from app.ingestion import MineruPdfParser, StructuredMineruChunker, build_doc_id
from app.ingestion.types import ChunkRecord, DocumentParser
from app.session import SQLiteSessionStore


DEFAULT_UPLOAD_DIR = "data/raw/uploads"
DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"
DEFAULT_MINERU_ARTIFACT_DIR = "data/processed/mineru"
JOB_RUNNING_STATUSES = {"queued", "running"}
PDF_EXTENSION_PATTERN = re.compile(r"\.pdf$", re.IGNORECASE)
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_pdf_name(file_name: str) -> str:
    base_name = Path(file_name or "document.pdf").name.strip() or "document.pdf"
    if not PDF_EXTENSION_PATTERN.search(base_name):
        base_name = f"{base_name}.pdf"
    return SAFE_FILENAME_PATTERN.sub("_", base_name)


def _load_chunks(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@dataclass
class DocumentJob:
    job_id: str
    status: str
    stage: str
    file_name: str
    upload_path: str
    doc_id: str | None = None
    doc_name: str | None = None
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "file_name": self.file_name,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class DocumentRecord:
    doc_id: str
    doc_name: str
    chunk_count: int | None = None


class DocumentManager:
    @classmethod
    def from_env(
        cls,
        *,
        retriever,
        session_store: SQLiteSessionStore | None = None,
    ) -> "DocumentManager":
        root = _project_root()
        return cls(
            retriever=retriever,
            session_store=session_store,
            upload_dir=root / DEFAULT_UPLOAD_DIR,
            chunks_path=root / DEFAULT_CHUNKS_PATH,
            artifact_dir=root / DEFAULT_MINERU_ARTIFACT_DIR,
        )

    def __init__(
        self,
        *,
        retriever,
        session_store: SQLiteSessionStore | None = None,
        upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
        chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
        artifact_dir: Path | str = DEFAULT_MINERU_ARTIFACT_DIR,
        parser_factory: Callable[[], DocumentParser] | None = None,
        max_chars: int = 1200,
    ):
        self.retriever = retriever
        self.session_store = session_store
        self.upload_dir = Path(upload_dir)
        self.chunks_path = Path(chunks_path)
        self.artifact_dir = Path(artifact_dir)
        self.parser_factory = parser_factory or self._default_parser_factory
        self.chunker = StructuredMineruChunker(max_chars=max_chars)
        self._jobs: dict[str, DocumentJob] = {}
        self._lock = Lock()

    def _default_parser_factory(self) -> DocumentParser:
        return MineruPdfParser.from_env(artifact_root=self.artifact_dir)

    def has_active_job(self) -> bool:
        with self._lock:
            return any(job.status in JOB_RUNNING_STATUSES for job in self._jobs.values())

    def list_jobs(self) -> list[DocumentJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)

    def get_job(self, job_id: str) -> DocumentJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def create_upload_job(self, file_name: str, content: bytes) -> DocumentJob:
        if not PDF_EXTENSION_PATTERN.search(file_name or ""):
            raise ValueError("Only PDF files can be uploaded")

        digest = hashlib.sha256(content).hexdigest()
        safe_name = _safe_pdf_name(file_name)
        upload_path = self.upload_dir / f"{digest[:12]}-{safe_name}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        existed_before = upload_path.exists()
        upload_path.write_bytes(content)

        now = _now()
        job = DocumentJob(
            job_id=str(uuid4()),
            status="queued",
            stage="queued",
            file_name=safe_name,
            upload_path=str(upload_path),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if any(existing.status in JOB_RUNNING_STATUSES for existing in self._jobs.values()):
                if not existed_before:
                    upload_path.unlink(missing_ok=True)
                raise RuntimeError("A document job is already running")
            self._jobs[job.job_id] = job
        return job

    def run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        parser = None
        try:
            upload_path = Path(job.upload_path)
            doc_id = build_doc_id(upload_path.resolve())
            self._set_job(job_id, status="running", stage="parsing", doc_id=doc_id, doc_name=upload_path.name)
            parser = self.parser_factory()
            parsed_document = parser.parse(upload_path)

            self._set_job(
                job_id,
                stage="chunking",
                doc_id=parsed_document.doc_id,
                doc_name=parsed_document.doc_name,
            )
            chunks = self.chunker.chunk(parsed_document)

            self._set_job(job_id, stage="embedding")
            embedded_chunks = self.retriever.embed_chunks(chunks)

            self._set_job(job_id, stage="indexing")
            delete_document = getattr(self.retriever, "delete_document", None)
            if callable(delete_document):
                delete_document(parsed_document.doc_id)
            self.retriever.upsert_embedded_chunks(embedded_chunks)
            self._merge_chunks(parsed_document.doc_id, chunks)
            self._invalidate_retriever_cache()

            self._set_job(job_id, status="succeeded", stage="done", error=None)
        except Exception as exc:
            self._set_job(job_id, status="failed", stage="failed", error=str(exc))
        finally:
            parser_close = getattr(parser, "close", None)
            if callable(parser_close):
                parser_close()

    def list_documents(self) -> list[DocumentRecord]:
        chunk_counts = self.chunk_counts()
        records = []
        for document in self.retriever.list_documents():
            records.append(
                DocumentRecord(
                    doc_id=document.doc_id,
                    doc_name=document.doc_name,
                    chunk_count=chunk_counts.get(document.doc_id),
                )
            )
        return records

    def chunk_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in _load_chunks(self.chunks_path):
            doc_id = str(chunk.get("doc_id", "")).strip()
            if not doc_id:
                continue
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def delete_document(self, doc_id: str) -> bool:
        resolved_doc_id = doc_id.strip()
        if not resolved_doc_id:
            return False
        if self.has_active_job():
            raise RuntimeError("A document job is already running")

        existed = self._document_exists(resolved_doc_id)
        self._remove_chunks(resolved_doc_id)
        delete_document = getattr(self.retriever, "delete_document", None)
        if callable(delete_document):
            delete_document(resolved_doc_id)
        shutil.rmtree(self.artifact_dir / resolved_doc_id, ignore_errors=True)
        if self.session_store is not None:
            self.session_store.clear_document_references(resolved_doc_id)
        self._invalidate_retriever_cache()
        return existed

    def _document_exists(self, doc_id: str) -> bool:
        if doc_id in self.chunk_counts():
            return True
        return any(document.doc_id == doc_id for document in self.retriever.list_documents())

    def _merge_chunks(self, doc_id: str, chunks: list[ChunkRecord]) -> None:
        existing = [
            chunk
            for chunk in _load_chunks(self.chunks_path)
            if chunk.get("doc_id") != doc_id
        ]
        _write_chunks(self.chunks_path, [*existing, *chunks])

    def _remove_chunks(self, doc_id: str) -> None:
        existing = [
            chunk
            for chunk in _load_chunks(self.chunks_path)
            if chunk.get("doc_id") != doc_id
        ]
        _write_chunks(self.chunks_path, existing)

    def _invalidate_retriever_cache(self) -> None:
        invalidate = getattr(self.retriever, "invalidate_cache", None)
        if callable(invalidate):
            invalidate()

    def _set_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = _now()
