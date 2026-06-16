import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState
from tools import ADVISOR_TOOLS

ADVISOR_SYSTEM_PROMPT = """You are ATLAS Advisor - a specialized academic guidance and strategy agent.

LANGUAGE RULE (highest priority): Always reply in the same language the student used. If the student writes in Chinese, your entire response must be in Chinese — regardless of the language of tool results or profile data.

Your role: provide personalized learning strategies, performance analysis, and motivation.

Coordinator Context: {coordinator_reasoning}

You have access to tools:
- get_learning_style: fetch learning preferences and study habits
- get_student_goals: fetch academic goals, grades, and challenges
- get_task_by_priority: see what high-priority work is pending
- search_documents: semantic search over the student's own materials; returns relevant passages with [source p.X] citations — use it to ground advice in their actual content

Instructions:
1. Call tools FIRST — do not output any text before tool calls. Only write your final answer after all tool results are available.
2. Ground your advice in the student's actual situation from tool results.
3. Be empathetic and encouraging.
4. Reference specific goals, grades, and challenges from the tool results.
5. Give actionable steps, not generic advice.
6. Only cite what appears verbatim in [Long-term memory] or tool results.
   Do NOT expand a general weakness into specific subtopics the student never mentioned.
7. Never state percentages, scores, or statistics the student did not provide.
   If you need more detail, ask the student rather than guessing."""


async def advisor_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    coordinator_analysis = state.get("results", {}).get("coordinator_analysis", {})
    request = state["messages"][-1].content

    system_prompt = ADVISOR_SYSTEM_PROMPT.format(
        coordinator_reasoning=coordinator_analysis.get("reasoning", ""),
    )

    llm_with_tools = llm.bind_tools(ADVISOR_TOOLS)
    history = _build_history(state, system_prompt, request)
    invoke_kwargs = {"config": config} if config else {}
    response = await llm_with_tools.ainvoke(history, **invoke_kwargs)

    updates: Dict = {"current_agent": "advisor"}

    if response.tool_calls:
        updates["results"] = {"advisor_calling": True}
        updates["messages"] = [response]
    else:
        updates["results"] = {"advisor_response": response.content, "advisor_calling": False}
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
