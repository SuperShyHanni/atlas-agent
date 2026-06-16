"""RAG retrieval tool — semantic search over the student's ingested study
materials. Returns only the relevant chunks with source/page citations, instead
of dumping a whole file into the prompt.
"""
from langchain_core.tools import tool

from .context import get_state
from vector_store import search_documents as _vs_search_documents


@tool
def search_documents(query: str) -> str:
    """
    Search the student's uploaded study materials (PDFs, lecture notes, files)
    for passages relevant to a question, and return them with source citations.
    Use this instead of read_file when you need a specific fact or explanation
    from the student's documents rather than the whole file.

    Args:
        query: What to look for, phrased as the concept or question to retrieve.
    """
    user_id = get_state().get("session_id", "")
    if not user_id:
        return "No active session; cannot search documents."

    hits = _vs_search_documents(user_id, query, k=4)
    if not hits:
        return "No relevant passages found in the student's documents (none ingested yet?)."

    blocks = []
    for h in hits:
        cite = h["source"] + (f" p.{h['page']}" if h.get("page") else "")
        blocks.append(f"[{cite}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)
