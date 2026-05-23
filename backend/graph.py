import os
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import AcademicState
from agents.coordinator import coordinator_node
from agents.planner import planner_node
from agents.notewriter import notewriter_node
from agents.advisor import advisor_node
from tools import PLANNER_TOOLS, NOTEWRITER_TOOLS, ADVISOR_TOOLS
import tools.calendar_tools as cal_mod
import tools.task_tools as task_mod
import tools.profile_tools as profile_mod


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "ep-20260329203325-lg8tt"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        streaming=True,
    )


def inject_state_into_tools(state: AcademicState):
    """Inject current session state into tool modules so tools can read live data."""
    cal_mod.set_state(state)
    task_mod.set_state(state)
    profile_mod.set_state(state)


def route_after_coordinator(state: AcademicState) -> str:
    analysis = state.get("results", {}).get("coordinator_analysis", {})
    required = analysis.get("required_agents", ["ADVISOR"])
    priority = analysis.get("priority_agent", required[0] if required else "ADVISOR")
    agent_map = {"PLANNER": "planner", "NOTEWRITER": "notewriter", "ADVISOR": "advisor"}
    return agent_map.get(priority.upper(), "advisor")


def should_use_tools(state: AcademicState) -> str:
    """After an agent runs, check if the last message has tool calls."""
    messages = state.get("messages", [])
    if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
        return "tools"
    return END


def make_node(fn: Callable, llm: ChatOpenAI) -> Callable:
    async def wrapped(state: AcademicState, config: RunnableConfig):
        inject_state_into_tools(state)
        return await fn(state, llm, config)
    return wrapped


def build_graph():
    llm = get_llm()

    all_tools = list({t.name: t for t in PLANNER_TOOLS + NOTEWRITER_TOOLS + ADVISOR_TOOLS}.values())
    tool_node = ToolNode(all_tools)

    builder = StateGraph(AcademicState)

    builder.add_node("coordinator", make_node(coordinator_node, llm))
    builder.add_node("planner", make_node(planner_node, llm))
    builder.add_node("notewriter", make_node(notewriter_node, llm))
    builder.add_node("advisor", make_node(advisor_node, llm))
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {"planner": "planner", "notewriter": "notewriter", "advisor": "advisor"},
    )

    # Each agent: if it emits tool calls → tools node, else → END
    for agent in ("planner", "notewriter", "advisor"):
        builder.add_conditional_edges(agent, should_use_tools, {"tools": "tools", END: END})

    # After tools execute → route back to whichever agent called them
    def route_tools_back(state: AcademicState) -> str:
        results = state.get("results", {})
        if "planner_calling" in results:
            return "planner"
        if "notewriter_calling" in results:
            return "notewriter"
        return "advisor"

    builder.add_conditional_edges(
        "tools",
        route_tools_back,
        {"planner": "planner", "notewriter": "notewriter", "advisor": "advisor"},
    )

    return builder.compile()


graph = build_graph()
