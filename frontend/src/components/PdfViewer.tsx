import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as pdfjsLib from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";

import {
  documentPdfUrl,
  type DocumentPageResponse,
  type DocumentResponse,
} from "../api/client";
import { Icon } from "./Icon";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

type PdfDocument = Awaited<ReturnType<typeof pdfjsLib.getDocument>["promise"]>;

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
  const [pdfDoc, setPdfDoc] = useState<PdfDocument | null>(null);
  const [pageHeights, setPageHeights] = useState<number[]>([]);
  const [renderRange, setRenderRange] = useState({ start: 0, end: 0 });
  const [loadError, setLoadError] = useState("");
  const scaleRef = useRef(1);

  useEffect(() => {
    let cancelled = false;
    setPdfDoc(null);
    setPageHeights([]);
    setRenderRange({ start: 0, end: 0 });
    setLoadError("");
    const loadingTask = pdfjsLib.getDocument(documentPdfUrl(docId));

    loadingTask.promise.then(async (pdf) => {
      if (cancelled) return;
      const container = containerRef.current;
      if (!container) return;

      const availableWidth = Math.max(280, container.clientWidth - 24);
      const heights: number[] = [];

      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        if (cancelled) return;
        const viewport = page.getViewport({ scale: 1 });
        const scale = Math.min(1.7, Math.max(0.55, availableWidth / viewport.width));
        scaleRef.current = scale;
        heights.push(page.getViewport({ scale }).height);
      }

      if (!cancelled) {
        setPdfDoc(pdf);
        setPageHeights(heights);
      }
    }).catch((error: unknown) => {
      if (!cancelled) {
        setLoadError(error instanceof Error ? error.message : "PDF 加载失败");
      }
    });

    return () => {
      cancelled = true;
      void loadingTask.destroy();
    };
  }, [docId]);

  const totalHeight = useMemo(() => {
    if (!pageHeights.length) return 0;
    return pageHeights.reduce((sum, h) => sum + h, 0) + (pageHeights.length - 1) * PAGE_GAP;
  }, [pageHeights]);

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

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !pageHeights.length) return;

    const handleScroll = () => {
      const viewCenter = container.scrollTop + container.clientHeight / 2;
      let offset = 0;
      for (let i = 0; i < pageHeights.length; i++) {
        const pageBottom = offset + pageHeights[i];
        if (viewCenter >= offset && viewCenter <= pageBottom) {
          onVisiblePageChange(i + 1);
          break;
        }
        offset = pageBottom + PAGE_GAP;
      }
    };

    container.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => container.removeEventListener("scroll", handleScroll);
  }, [pageHeights, onVisiblePageChange]);

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

  useEffect(() => {
    if (!scrollToPage || !pageHeights.length || !containerRef.current) return;
    const targetPage = Math.min(Math.max(1, scrollToPage), pageHeights.length);
    containerRef.current.scrollTo({ top: getPageOffset(targetPage), behavior: "smooth" });
  }, [scrollToPage, pageHeights, getPageOffset]);

  return (
    <div className="pdf-scroll-container" ref={containerRef}>
      {loadError ? <p className="pdf-error">{loadError}</p> : null}
      <div className="pdf-scroll-spacer" style={{ height: totalHeight, position: "relative" }}>
        {pdfDoc && pageHeights.length > 0 &&
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
                <PdfPageCanvas pdf={pdfDoc} page={pageNum} scale={scaleRef.current} />
                <span className="pdf-page-number">{pageNum}</span>
              </div>
            );
          })}
      </div>
    </div>
  );
}

function PdfPageCanvas({ pdf, page, scale }: { pdf: PdfDocument; page: number; scale: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let renderTask: { cancel: () => void; promise: Promise<unknown> } | null = null;

    async function renderPage() {
      const canvas = canvasRef.current;
      if (!canvas) return;

      setError("");
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
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);

      renderTask = pdfPage.render({
        canvas,
        viewport,
        transform: pixelRatio === 1 ? undefined : [pixelRatio, 0, 0, pixelRatio, 0, 0],
      });
      await renderTask.promise;
    }

    renderPage().catch((err: unknown) => {
      if (cancelled) return;
      setError(err instanceof Error ? err.message : "PDF 渲染失败");
    });

    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [pdf, page, scale]);

  return (
    <div className="pdf-canvas-wrap">
      <canvas ref={canvasRef} />
      {error ? <p className="pdf-error">{error}</p> : null}
    </div>
  );
}

export function DocumentPanel({
  documents,
  activeDocId,
  scrollToPage,
  pageDetail,
  collapsed,
  onToggle,
  onOpenPage,
  onVisiblePageChange,
  onUploadDocument,
  onDeleteDocument,
  onAskCurrentPage,
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
  onUploadDocument: (file: File) => void;
  onDeleteDocument: (docId: string) => void;
  onAskCurrentPage: () => void;
  onResize: (width: number) => void;
  onResizeStart: () => void;
  onResizeEnd: () => void;
}) {
  const handleRef = useRef<HTMLDivElement | null>(null);
  const uploadRef = useRef<HTMLInputElement | null>(null);
  const activeDoc = documents.find((doc) => doc.id === activeDocId) ?? documents[0] ?? null;
  const [pageInput, setPageInput] = useState("1");

  useEffect(() => {
    setPageInput(String(pageDetail?.page ?? 1));
  }, [pageDetail?.page, activeDoc?.id]);

  useEffect(() => {
    const handle = handleRef.current;
    if (!handle || collapsed) return;

    let startX = 0;
    let startWidth = 0;

    const onMouseMove = (e: MouseEvent) => {
      const delta = startX - e.clientX;
      onResize(Math.min(900, Math.max(280, startWidth + delta)));
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

  const jumpToPage = () => {
    if (!activeDoc) return;
    const page = Math.min(activeDoc.page_count || 1, Math.max(1, Number.parseInt(pageInput, 10) || 1));
    onOpenPage(activeDoc.id, page);
    setPageInput(String(page));
  };

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
                {doc.name}{doc.parsed ? "" : "（未解析）"}
              </option>
            ))}
          </select>
        </label>
        <div className="document-action-row">
          <input
            ref={uploadRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onUploadDocument(file);
              event.currentTarget.value = "";
            }}
          />
          <button type="button" onClick={() => uploadRef.current?.click()}>
            <Icon name="upload" /> 上传
          </button>
          <button type="button" disabled={!activeDoc} onClick={() => activeDoc && onDeleteDocument(activeDoc.id)}>
            <Icon name="trash" /> 删除
          </button>
        </div>
        <div className="page-jump-row">
          <label>
            <span>页码</span>
            <input
              value={pageInput}
              inputMode="numeric"
              onChange={(event) => setPageInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") jumpToPage();
              }}
            />
          </label>
          <button type="button" disabled={!activeDoc} onClick={jumpToPage}>跳转</button>
          <button type="button" disabled={!activeDoc?.parsed} onClick={onAskCurrentPage}>问当前页</button>
        </div>
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
          {activeDoc ? <span>p.{pageDetail?.page ?? "-"} / {activeDoc.page_count || "-"}</span> : null}
        </div>
        {!activeDoc?.parsed ? <p>PDF 已上传，但还没有 MinerU 解析结果。解析后即可读取页面并用于问答。</p> : <p>{pageDetail?.text || "滚动 PDF 查看内容，或点击引用跳转到指定页面。"}</p>}
      </div>
    </aside>
  );
}
