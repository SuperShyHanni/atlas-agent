"""Document RAG ingestion — read a study file from FILES_DIR, split into
overlapping chunks (per page for PDFs, so chunks carry page citations), and embed
into the `atlas_documents` Chroma collection.

This replaces dumping whole files into the prompt via `read_file`: instead the
agent retrieves only the relevant chunks (cheaper, more accurate, citable).
"""
from pathlib import Path
from typing import List, Optional, Tuple

from tools.filesystem_tools import _get_files_dir, _ALLOWED_EXTENSIONS
from vector_store import add_document_chunks

CHUNK_SIZE = 200       # characters — tuned via eval/evaluate_rag.py sweep
CHUNK_OVERLAP = 40


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping windows, preferring paragraph/sentence breaks."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # back off to the nearest break so we don't cut mid-sentence
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", "。", " "):
                idx = window.rfind(sep)
                if idx > size * 0.5:
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract(target: Path) -> List[Tuple[Optional[int], str]]:
    """Return [(page_no, text)] — page_no is set only for PDFs."""
    suffix = target.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(target))
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append((i + 1, txt))
        return pages
    return [(None, target.read_text(encoding="utf-8", errors="replace"))]


def ingest_file(user_id: str, filename: str) -> dict:
    """Ingest one file from the student's FILES_DIR into the RAG store."""
    base = _get_files_dir()
    target = (base / filename).resolve()

    # Safety: same path-traversal guard as the filesystem tools
    if not str(target).startswith(str(base)):
        return {"error": "Access denied: path is outside the allowed directory."}
    if not target.exists():
        return {"error": f"File not found: {filename}"}
    if target.suffix.lower() not in _ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type: {target.suffix}"}

    chunks: List[dict] = []
    for page_no, text in _extract(target):
        for piece in chunk_text(text):
            chunks.append({"text": piece, "page": page_no})

    count = add_document_chunks(user_id, filename, chunks)
    return {"source": filename, "chunks_indexed": count}
