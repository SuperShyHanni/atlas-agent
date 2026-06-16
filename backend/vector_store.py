"""
L3 Long-term Memory — ChromaDB vector store.

Stores key facts extracted from conversations as embeddings.
Retrieval is semantic: a new question triggers similarity search
to surface relevant past observations about the student.
"""
import math
import os
import re
import uuid

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

_client = None
_collection = None
_doc_collection = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = _get_client().get_or_create_collection(
            name="atlas_memories",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_doc_collection():
    """Separate collection for RAG over the student's study materials (PDFs, notes).
    Kept distinct from `atlas_memories` (conversation facts) — different content,
    different retrieval semantics."""
    global _doc_collection
    if _doc_collection is None:
        ef = embedding_functions.DefaultEmbeddingFunction()
        _doc_collection = _get_client().get_or_create_collection(
            name="atlas_documents",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _doc_collection


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


# ── Document RAG ──────────────────────────────────────────────────────────────

def add_document_chunks(user_id: str, source: str, chunks: list) -> int:
    """Embed and store chunks of a document. Re-ingesting a source replaces its
    old chunks. `chunks` is a list of {"text": str, "page": int|None}."""
    col = _get_doc_collection()

    # Replace any existing chunks for this (user, source) so re-ingest is clean
    delete_user_documents(user_id, source=source)

    ids, docs, metas = [], [], []
    for i, ch in enumerate(chunks):
        text = (ch.get("text") or "").strip()
        if not text:
            continue
        ids.append(f"{user_id}::{source}::{i}::{uuid.uuid4().hex[:8]}")
        docs.append(text)
        meta = {"user_id": user_id, "source": source, "chunk_index": i}
        if ch.get("page") is not None:
            meta["page"] = ch["page"]
        metas.append(meta)

    if not docs:
        return 0
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _bm25_scores(query: str, docs: list, k1: float = 1.5, b: float = 0.75) -> list:
    """Classic BM25 keyword relevance over `docs`; higher = more relevant.
    Gives lexical/acronym matching (BCNF, TCP, Dijkstra) that pure embeddings miss."""
    toks = [_tokenize(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks) / n
    df: dict = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    qtok = set(_tokenize(query))
    scores = [0.0] * n
    for i, t in enumerate(toks):
        if not t:
            continue
        tf: dict = {}
        for w in t:
            tf[w] = tf.get(w, 0) + 1
        dl = len(t)
        s = 0.0
        for w in qtok:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def _rrf(rankings: list, weights: list = None, c: int = 60) -> dict:
    """Weighted Reciprocal Rank Fusion of ranked id-lists → fused score per id."""
    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict = {}
    for w, ranked_ids in zip(weights, rankings):
        for rank, _id in enumerate(ranked_ids):
            fused[_id] = fused.get(_id, 0.0) + w / (c + rank + 1)
    return fused


# Tuned on the eval set (see eval/evaluate_rag.py sweep): dense embeddings are the
# stronger signal, so the vector ranking gets 0.8 of the fusion weight vs BM25's 0.2.
HYBRID_VEC_WEIGHT = 0.8


def search_documents(user_id: str, query: str, k: int = 4, source: str = "",
                     mode: str = "hybrid", vec_weight: float = HYBRID_VEC_WEIGHT) -> list:
    """Return top-k relevant chunks for this user, each with citation metadata.

    mode:
      "vector" — embedding similarity only (baseline)
      "bm25"   — lexical only
      "hybrid" — weighted Reciprocal Rank Fusion of vector + BM25 (default; best on eval)
    """
    col = _get_doc_collection()
    where = {"user_id": user_id}
    if source:
        where = {"$and": [{"user_id": user_id}, {"source": source}]}

    existing = col.get(where=where, include=["documents", "metadatas"])
    ids = existing.get("ids", [])
    if not ids:
        return []
    all_docs = existing.get("documents", [])
    all_metas = existing.get("metadatas", [])
    meta_by_id = dict(zip(ids, zip(all_docs, all_metas)))

    # Vector ranking over the full candidate set
    vec_ids: list = []
    if mode in ("vector", "hybrid"):
        res = col.query(
            query_texts=[query],
            n_results=len(ids),
            where=where,
            include=["distances"],
        )
        vec_ids = res.get("ids", [[]])[0]

    # BM25 ranking
    bm25_ids: list = []
    if mode in ("bm25", "hybrid"):
        scores = _bm25_scores(query, all_docs)
        bm25_ids = [ids[i] for i in sorted(range(len(ids)), key=lambda j: scores[j], reverse=True)]

    if mode == "vector":
        ordered = vec_ids
    elif mode == "bm25":
        ordered = bm25_ids
    else:
        fused = _rrf([vec_ids, bm25_ids], weights=[vec_weight, 1.0 - vec_weight])
        ordered = sorted(fused, key=lambda i: fused[i], reverse=True)

    out = []
    for _id in ordered[:k]:
        doc, meta = meta_by_id.get(_id, ("", {}))
        out.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "page": meta.get("page"),
        })
    return out


def list_document_sources(user_id: str) -> list:
    """Distinct source filenames this user has ingested."""
    existing = _get_doc_collection().get(where={"user_id": user_id})
    sources = {m.get("source") for m in existing.get("metadatas", []) if m.get("source")}
    return sorted(sources)


def delete_user_documents(user_id: str, source: str = ""):
    col = _get_doc_collection()
    if source:
        col.delete(where={"$and": [{"user_id": user_id}, {"source": source}]})
    else:
        col.delete(where={"user_id": user_id})
