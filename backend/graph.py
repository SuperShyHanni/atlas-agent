import os
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from state import AcademicState
from agents.coordinator import coordinator_node
from agents.planner import planner_node
from agents.notewriter import notewriter_node
from agents.advisor import advisor_node


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "ep-20260329203325-lg8tt"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        streaming=True,
    )


def route_after_coordinator(state: AcademicState) -> str:
    analysis = state.get("results", {}).get("coordinator_analysis", {})
    required = analysis.get("required_agents", ["ADVISOR"])
    priority = analysis.get("priority_agent", required[0] if required else "ADVISOR")
    agent_map = {"PLANNER": "planner", "NOTEWRITER": "notewriter", "ADVISOR": "advisor"}
    return agent_map.get(priority.upper(), "advisor")


def make_node(fn: Callable, llm: ChatOpenAI) -> Callable:
    async def wrapped(state: AcademicState, config: RunnableConfig):
        return await fn(state, llm, config)
    return wrapped


def build_graph():
    llm = get_llm()

    builder = StateGraph(AcademicState)
    builder.add_node("coordinator", make_node(coordinator_node, llm))
    builder.add_node("planner", make_node(planner_node, llm))
    builder.add_node("notewriter", make_node(notewriter_node, llm))
    builder.add_node("advisor", make_node(advisor_node, llm))

    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {"planner": "planner", "notewriter": "notewriter", "advisor": "advisor"},
    )
    builder.add_edge("planner", END)
    builder.add_edge("notewriter", END)
    builder.add_edge("advisor", END)

    return builder.compile()


graph = build_graph()
