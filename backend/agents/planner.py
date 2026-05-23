import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState
from tools import PLANNER_TOOLS

PLANNER_SYSTEM_PROMPT = """You are ATLAS Planner - a specialized academic scheduling and time management agent.

Your role: create optimized study schedules, manage deadlines, plan study sessions.

Student Profile:
{profile_summary}

Coordinator Context: {coordinator_reasoning}

You have access to tools to fetch real-time data:
- search_calendar: find upcoming classes, exams, and events
- get_upcoming_deadlines: get tasks and exams sorted by urgency
- get_pending_tasks: list incomplete assignments
- get_task_by_priority: filter tasks by priority level

Instructions:
1. Use tools to gather current schedule and task data BEFORE making recommendations
2. Apply ReACT reasoning after getting tool results:
   - Thought: What does the data tell me about the student's situation?
   - Action: What scheduling decisions follow from this data?
   - Plan: Concrete time blocks with specific dates and times
3. Always reference actual calendar events and task deadlines in your response
4. Consider the student's peak study hours and learning style
5. Format output in clear markdown with a weekly schedule table"""


async def planner_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    profile = state.get("profile", {})
    request = state["messages"][-1].content
    coordinator_analysis = state.get("results", {}).get("coordinator_analysis", {})

    personal_info = profile.get("personal_info", {})
    prefs = profile.get("learning_preferences", {})

    profile_summary = json.dumps({
        "name": personal_info.get("name", "Student"),
        "major": personal_info.get("major"),
        "year": personal_info.get("academic_year"),
        "learning_style": prefs.get("learning_style"),
        "peak_hours": prefs.get("peak_study_hours"),
        "study_hours_per_day": prefs.get("study_hours_per_day"),
    }, indent=2)

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        profile_summary=profile_summary,
        coordinator_reasoning=coordinator_analysis.get("reasoning", ""),
    )

    llm_with_tools = llm.bind_tools(PLANNER_TOOLS)

    # Build message history: system + prior tool exchanges + current request
    history = _build_history(state, system_prompt, request)
    invoke_kwargs = {"config": config} if config else {}
    response = await llm_with_tools.ainvoke(history, **invoke_kwargs)

    updates: Dict = {"current_agent": "planner"}

    if response.tool_calls:
        # Signal which agent should resume after tools execute
        updates["results"] = {"planner_calling": True}
        updates["messages"] = [response]
    else:
        updates["results"] = {"planner_response": response.content, "planner_calling": False}
        updates["messages"] = [response]

    return updates


def _build_history(state: AcademicState, system_prompt: str, request: str) -> list:
    """Reconstruct message history including any prior tool call rounds."""
    messages = [SystemMessage(content=system_prompt)]
    # Include prior tool exchanges stored in state messages (after the user request)
    for m in state.get("messages", []):
        if hasattr(m, "tool_calls") or m.__class__.__name__ == "ToolMessage":
            messages.append(m)
    messages.append(HumanMessage(content=request))
    return messages
