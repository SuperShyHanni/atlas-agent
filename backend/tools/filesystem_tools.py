"""
Filesystem tools — list and read files (including PDFs) from a configured directory.
Requires FILES_DIR in environment (defaults to ~/Desktop).
"""
import os
from pathlib import Path
from langchain_core.tools import tool

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".json"}


def _get_files_dir() -> Path:
    raw = os.getenv("FILES_DIR", str(Path.home() / "Desktop"))
    return Path(raw).expanduser().resolve()


@tool
def list_files(subdirectory: str = "") -> str:
    """
    List readable files in the student's files directory.
    Optionally pass a subdirectory name to narrow the listing.
    """
    base = _get_files_dir()
    target = (base / subdirectory).resolve() if subdirectory else base

    # Safety: prevent path traversal outside the allowed base
    if not str(target).startswith(str(base)):
        return "Access denied: path is outside the allowed directory."

    if not target.exists():
        return f"Directory not found: {target}"

    entries = []
    for p in sorted(target.iterdir()):
        if p.is_file() and p.suffix.lower() in _ALLOWED_EXTENSIONS:
            size_kb = p.stat().st_size // 1024
            entries.append(f"{p.name} ({size_kb} KB)")
        elif p.is_dir():
            entries.append(f"{p.name}/  (folder)")

    return "\n".join(entries) if entries else "No readable files found."


@tool
def read_file(filename: str) -> str:
    """
    Read the content of a file from the student's files directory.
    Supports .pdf, .txt, .md, .py, .json files.
    For PDFs, extracts text from all pages.
    """
    base = _get_files_dir()
    target = (base / filename).resolve()

    # Safety: prevent path traversal
    if not str(target).startswith(str(base)):
        return "Access denied: path is outside the allowed directory."

    if not target.exists():
        return f"File not found: {filename}"

    suffix = target.suffix.lower()

    if suffix not in _ALLOWED_EXTENSIONS:
        return f"File type '{suffix}' is not supported. Allowed: {', '.join(_ALLOWED_EXTENSIONS)}"

    if suffix == ".pdf":
        return _read_pdf(target)

    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages) if pages else "(PDF has no extractable text)"
    except Exception as e:
        return f"PDF read error: {e}"
