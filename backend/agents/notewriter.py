import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState
from tools import NOTEWRITER_TOOLS

NOTEWRITER_SYSTEM_PROMPT = """You are ATLAS NoteWriter - a specialized academic content creation agent.

Your role: create study notes, summaries, flashcards, and study guides.

Coordinator Context: {coordinator_reasoning}

You have access to tools:
- get_learning_style: fetch the student's learning style and preferences

Instructions:
1. Call get_learning_style first to understand how to tailor the content
2. Apply ReACT reasoning:
   - Thought: What format best suits this student's learning style?
   - Action: Structure content accordingly
   - Output: Well-formatted study material
3. Adapt format to learning style:
   - Visual → tables, ASCII diagrams, structured layouts
   - Auditory → narrative explanations, mnemonics
   - Reading/Writing → detailed notes with headers
   - Kinesthetic → examples, practice problems
4. Format in clear markdown"""


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
    for m in state.get("messages", []):
        if hasattr(m, "tool_calls") or m.__class__.__name__ == "ToolMessage":
            messages.append(m)
    messages.append(HumanMessage(content=request))
    return messages
