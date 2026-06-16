import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState
from tools import NOTEWRITER_TOOLS

NOTEWRITER_SYSTEM_PROMPT = """You are ATLAS NoteWriter - a specialized academic content creation agent.

LANGUAGE RULE (highest priority): Always reply in the same language the student used. If the student writes in Chinese, your entire response must be in Chinese — regardless of the language of tool results or profile data.

Your role: create study notes, summaries, flashcards, and study guides.

Coordinator Context: {coordinator_reasoning}

You have access to tools:
- get_learning_style: fetch the student's learning style and preferences
- list_files: list files (PDF, txt, md) in the student's files directory
- read_file: read the FULL content of a file or PDF (use only when you truly need the whole file)
- search_documents: semantic search over the student's ingested materials; returns only the relevant passages with [source p.X] citations. PREFER this over read_file when you need specific facts.
- notion_search: search existing pages in the student's Notion workspace
- notion_get_page: retrieve full content of a Notion page by ID or URL
- notion_create_page: save notes or study guides to the student's Notion workspace

Instructions:
1. Call tools FIRST — do not output any text before tool calls.
2. Always call get_learning_style to tailor the content format.
3. If the student asks about content in their materials, use search_documents to retrieve the relevant passages and cite the source/page. Use list_files + read_file only when the whole file is genuinely needed.
4. If the student asks to save notes to Notion, use notion_create_page after generating the content.
5. Adapt format to learning style:
   - Visual → tables, ASCII diagrams, structured layouts
   - Auditory → narrative explanations, mnemonics
   - Reading/Writing → detailed notes with headers
   - Kinesthetic → examples, practice problems
6. Format in clear markdown."""


async def notewriter_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    coordinator_analysis = state.get("results", {}).get("coordinator_analysis", {})
    request = state["messages"][-1].content

    system_prompt = NOTEWRITER_SYSTEM_PROMPT.format(
        coordinator_reasoning=coordinator_analysis.get("reasoning", ""),
    )

    llm_with_tools = llm.bind_tools(NOTEWRITER_TOOLS)
    history = _build_history(state, system_prompt, request)
    invoke_kwargs = {"config": config} if config else {}
    response = await llm_with_tools.ainvoke(history, **invoke_kwargs)

    updates: Dict = {"current_agent": "notewriter"}

    if response.tool_calls:
        updates["results"] = {"notewriter_calling": True}
        updates["messages"] = [response]
    else:
        updates["results"] = {"notewriter_response": response.content, "notewriter_calling": False}
        updates["messages"] = [response]

    return updates


def _build_history(state: AcademicState, system_prompt: str, request: str) -> list:
    messages = [SystemMessage(content=system_prompt)]
    # main.py saves the current message to SQLite before calling load_full_context,
    # so it appears twice in state.messages. Keep only the first occurrence so the
    # sliding-window history is intact while avoiding duplicates.
    seen_request = False
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage) and m.content == request:
            if not seen_request:
                messages.append(m)
                seen_request = True
        else:
            messages.append(m)
    if not seen_request:
        messages.append(HumanMessage(content=request))
    return messages
