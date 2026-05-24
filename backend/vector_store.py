"""
L3 Long-term Memory — ChromaDB vector store.

Stores key facts extracted from conversations as embeddings.
Retrieval is semantic: a new question triggers similarity search
to surface relevant past observations about the student.
"""
import os
import uuid

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = client.get_or_create_collection(
            name="atlas_memories",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_memory(user_id: str, text: str) -> str:
    """Embed and store a fact. Returns the generated memory ID."""
    memory_id = str(uuid.uuid4())
    _get_collection().upsert(
        ids=[memory_id],
        documents=[text],
        metadatas=[{"user_id": user_id}],
    )
    return memory_id


def search_memories(user_id: str, query: str, k: int = 3) -> list:
    """Return top-k semantically similar memories for this user."""
    col = _get_collection()
    # Guard: collection must have at least 1 doc for this user
    existing = col.get(where={"user_id": user_id}, limit=1)
    if not existing["ids"]:
        return []
    results = col.query(
        query_texts=[query],
        n_results=min(k, len(existing["ids"])),
        where={"user_id": user_id},
    )
    docs = results.get("documents", [[]])[0]
    return [d for d in docs if d]


def delete_user_memories(user_id: str):
    _get_collection().delete(where={"user_id": user_id})
