from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from database import (
    append_message,
    delete_messages_up_to,
    get_all_messages,
    get_latest_summary,
    save_summary,
)

MAX_CONTEXT_MESSAGES = 20
SUMMARIZE_TRIGGER = 30  # summarize when total exceeds this


async def load_context_messages(user_id: str) -> List:
    """
    Return LangChain message objects for the current context window.
    If history exceeds MAX_CONTEXT_MESSAGES, prepend the latest summary
    and return only the most recent MAX_CONTEXT_MESSAGES messages.
    """
    rows = await get_all_messages(user_id)
    lc_messages = _rows_to_lc(rows)

    if len(rows) <= MAX_CONTEXT_MESSAGES:
        return lc_messages

    recent = lc_messages[-MAX_CONTEXT_MESSAGES:]
    summary = await get_latest_summary(user_id)
    if summary:
        prefix = SystemMessage(content=f"[Earlier conversation summary]: {summary}")
        return [prefix] + recent
    return recent


async def save_user_message(user_id: str, content: str) -> int:
    return await append_message(user_id, "user", content)


async def save_assistant_message(user_id: str, content: str) -> int:
    return await append_message(user_id, "assistant", content)


async def maybe_summarize(user_id: str, llm) -> None:
    """Summarize and trim old messages when total exceeds SUMMARIZE_TRIGGER."""
    rows = await get_all_messages(user_id)
    if len(rows) <= SUMMARIZE_TRIGGER:
        return

    # Summarize all but the most recent MAX_CONTEXT_MESSAGES
    to_summarize = rows[:-MAX_CONTEXT_MESSAGES]
    if not to_summarize:
        return

    existing_summary = await get_latest_summary(user_id)
    transcript = "\n".join(f"{r['role'].upper()}: {r['content']}" for r in to_summarize)

    prompt = f"""Summarize the following academic assistant conversation concisely (3-5 sentences), preserving key facts about the student's goals, courses, and important decisions.

{"Previous summary to incorporate: " + existing_summary if existing_summary else ""}

Conversation:
{transcript}

Summary:"""

    result = await llm.ainvoke([HumanMessage(content=prompt)])
    summary_text = result.content.strip()

    last_id = to_summarize[-1]["id"]
    await save_summary(user_id, summary_text, last_id)
    await delete_messages_up_to(user_id, last_id)


def _rows_to_lc(rows: List[dict]) -> List:
    out = []
    for r in rows:
        if r["role"] == "user":
            out.append(HumanMessage(content=r["content"]))
        elif r["role"] == "assistant":
            out.append(AIMessage(content=r["content"]))
    return out
