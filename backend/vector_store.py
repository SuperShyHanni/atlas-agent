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


# Cosine distance below this threshold is treated as a near-duplicate / update to an existing fact.
# ChromaDB cosine distance: 0.0 = identical, ~0.15 = same sentence rephrased, 1.0 = orthogonal.
_DEDUP_DISTANCE = 0.15


def add_memory(user_id: str, text: str) -> str:
    """Embed and store a fact, replacing any near-duplicate that already exists."""
    col = _get_collection()

    # Fetch all IDs for this user (no limit — needed for correct n_results capping below)
    existing = col.get(where={"user_id": user_id})
    if existing["ids"]:
        # Find near-duplicates and remove them so we don't accumulate conflicting versions
        results = col.query(
            query_texts=[text],
            n_results=min(5, len(existing["ids"])),
            where={"user_id": user_id},
            include=["distances"],
        )
        ids_to_delete = [
            doc_id
            for doc_id, dist in zip(
                results.get("ids", [[]])[0],
                results.get("distances", [[]])[0],
            )
            if dist < _DEDUP_DISTANCE
        ]
        if ids_to_delete:
            col.delete(ids=ids_to_delete)

    memory_id = str(uuid.uuid4())
    col.upsert(
        ids=[memory_id],
        documents=[text],
        metadatas=[{"user_id": user_id}],
    )
    return memory_id


def search_memories(user_id: str, query: str, k: int = 3) -> list:
    """Return top-k semantically similar memories for this user."""
    col = _get_collection()
    # Fetch all IDs without limit so n_results is capped correctly (not always 1)
    existing = col.get(where={"user_id": user_id})
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
