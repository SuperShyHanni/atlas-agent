"""
Three-layer memory system:

  L1  Working memory    — current context window (last MAX_CONTEXT_MESSAGES msgs)
  L2  Short-term memory — sliding window + LLM summary stored in SQLite
  L3  Long-term memory  — key facts embedded in ChromaDB, recalled by similarity
"""
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import database as db
from vector_store import add_memory, delete_user_memories, search_memories

MAX_CONTEXT_MESSAGES = 20   # L1 window size
SUMMARIZE_TRIGGER = 30      # L2: summarize when total exceeds this


# ── L3: Long-term memory ──────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are extracting long-term memory facts from a student conversation.
From the exchange below, write 1-3 short bullet points capturing durable facts about the student:
preferences, goals, struggles, learning insights, or study habits worth remembering long-term.
Skip generic or one-off remarks. Output only the bullet points, nothing else.

User: {user_msg}
Assistant: {assistant_msg}

Facts (or "none" if nothing worth remembering):"""


async def extract_and_store_memory(
    user_id: str, user_msg: str, assistant_msg: str, llm
) -> None:
    """Extract key facts from a conversation exchange and embed them into ChromaDB."""
    prompt = _EXTRACT_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg[:800])
    result = await llm.ainvoke([HumanMessage(content=prompt)])
    text = result.content.strip()

    if text.lower() == "none" or not text:
        return

    # Each bullet becomes a separate embedding for finer retrieval
    for line in text.splitlines():
        line = line.lstrip("-• ").strip()
        if len(line) > 10:
            add_memory(user_id, line)


async def retrieve_relevant_memories(user_id: str, query: str, k: int = 3) -> List[str]:
    """Semantic search over ChromaDB for this user's long-term facts."""
    return search_memories(user_id, query, k=k)


async def clear_long_term_memory(user_id: str) -> None:
    delete_user_memories(user_id)


# ── L2: Short-term memory (summary) ──────────────────────────────────────────

_SUMMARY_PROMPT = """Summarize the following academic assistant conversation concisely (3-5 sentences),
preserving key facts about the student's goals, courses, and important decisions.
{prior}

Conversation:
{transcript}

Summary:"""


async def maybe_summarize(user_id: str, llm) -> None:
    """When history exceeds SUMMARIZE_TRIGGER, compress old messages into a summary (L2)."""
    rows = await db.get_all_messages(user_id)
    if len(rows) <= SUMMARIZE_TRIGGER:
        return

    to_summarize = rows[:-MAX_CONTEXT_MESSAGES]
    if not to_summarize:
        return

    existing = await db.get_latest_summary(user_id)
    transcript = "\n".join(
        f"{r['role'].upper()}: {r['content']}" for r in to_summarize
    )
    prior_line = f"Previous summary to incorporate: {existing}" if existing else ""

    prompt = _SUMMARY_PROMPT.format(prior=prior_line, transcript=transcript)
    result = await llm.ainvoke([HumanMessage(content=prompt)])

    last_id = to_summarize[-1]["id"]
    await db.save_summary(user_id, result.content.strip(), last_id)
    await db.delete_messages_up_to(user_id, last_id)


# ── L1 + L2 + L3: Full context assembly ──────────────────────────────────────

async def load_full_context(user_id: str, current_message: str) -> List:
    """
    Build the complete context list injected into LangGraph:

      [SystemMessage: L3 long-term memories]   ← semantic recall from ChromaDB
      [SystemMessage: L2 summary]               ← only when history was compressed
      [HumanMessage / AIMessage × ≤20]          ← L1 recent messages from SQLite
    """
    context: List = []

    # L3 — semantic recall
    memories = await retrieve_relevant_memories(user_id, current_message)
    if memories:
        bullet_list = "\n".join(f"- {m}" for m in memories)
        context.append(
            SystemMessage(
                content=f"[Long-term memory — known facts about this student]:\n{bullet_list}"
            )
        )

    # L2 — conversation summary (if history was compressed)
    rows = await db.get_all_messages(user_id)
    lc_messages = _rows_to_lc(rows)

    if len(rows) > MAX_CONTEXT_MESSAGES:
        summary = await db.get_latest_summary(user_id)
        if summary:
            context.append(
                SystemMessage(content=f"[Earlier conversation summary]: {summary}")
            )
        context.extend(lc_messages[-MAX_CONTEXT_MESSAGES:])
    else:
        # L1 — all recent messages fit in context
        context.extend(lc_messages)

    return context


# ── DB helpers ────────────────────────────────────────────────────────────────

async def save_user_message(user_id: str, content: str) -> int:
    return await db.append_message(user_id, "user", content)


async def save_assistant_message(user_id: str, content: str) -> int:
    return await db.append_message(user_id, "assistant", content)


def _rows_to_lc(rows: List[dict]) -> List:
    out = []
    for r in rows:
        if r["role"] == "user":
            out.append(HumanMessage(content=r["content"]))
        elif r["role"] == "assistant":
            out.append(AIMessage(content=r["content"]))
    return out
