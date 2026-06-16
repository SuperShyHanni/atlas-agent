"""Tool-level RBAC + audit for the agent ReAct loop.

Two-layer authorization:
  1. Binding layer (in each agent node): the LLM is only `bind_tools(...)`-ed with
     its own toolset, so it cannot *see* tools outside its role.
  2. Enforcement layer (here): even if a malformed/hallucinated tool call slips
     through, the guarded tool node checks the calling agent against an allow-list
     and refuses any cross-role (privilege-escalation) call before it executes.

Every call — allowed, denied or errored — is written to an immutable audit log
(`tool_audit`) keyed by user/session, agent, tool, args, status, latency, so the
whole ReAct tool trace is observable and traceable.
"""
import time
from typing import Callable, Dict, List, Set

from langchain_core.messages import ToolMessage

import database as db
from tools import PLANNER_TOOLS, NOTEWRITER_TOOLS, ADVISOR_TOOLS

# Graph-node name -> set of tool names that role is authorized to invoke.
AGENT_TOOL_POLICY: Dict[str, Set[str]] = {
    "planner": {t.name for t in PLANNER_TOOLS},
    "notewriter": {t.name for t in NOTEWRITER_TOOLS},
    "advisor": {t.name for t in ADVISOR_TOOLS},
}


def is_allowed(agent: str, tool_name: str) -> bool:
    return tool_name in AGENT_TOOL_POLICY.get(agent, set())


def make_guarded_tool_node(tools: List) -> Callable:
    """Build an async LangGraph node that executes tool calls under RBAC + audit."""
    registry = {t.name: t for t in tools}

    async def guarded_tool_node(state: dict) -> dict:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) or []
        agent = state.get("current_agent", "unknown")
        user_id = state.get("session_id", "unknown")
        run_id = state.get("run_id", "")

        out: List[ToolMessage] = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", "")
            started = time.perf_counter()

            if not is_allowed(agent, name):
                content = (
                    f"⛔ Permission denied: agent '{agent}' is not authorized to call "
                    f"tool '{name}'. This call was blocked and logged."
                )
                await _audit(user_id, run_id, agent, name, args, "denied", content, 0)
                out.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                continue

            tool = registry.get(name)
            if tool is None:
                content = f"Error: unknown tool '{name}'."
                await _audit(user_id, run_id, agent, name, args, "error", content, 0)
                out.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
                continue

            # Idempotent resume: if this exact call already succeeded earlier in
            # this run (e.g. before a crash), replay the stored result instead of
            # re-executing — prevents duplicate side effects (e.g. notion_create_page).
            cached = await _find_cached(run_id, name, args)
            if cached is not None:
                out.append(ToolMessage(content=cached, tool_call_id=call_id, name=name))
                continue

            try:
                result = await tool.ainvoke(args)
                status = "allowed"
            except Exception as e:  # tool runtime failure — still audited
                result = f"Error executing {name}: {e}"
                status = "error"

            latency_ms = int((time.perf_counter() - started) * 1000)
            await _audit(user_id, run_id, agent, name, args, status, str(result), latency_ms)
            out.append(ToolMessage(content=str(result), tool_call_id=call_id, name=name))

        return {"messages": out}

    return guarded_tool_node


async def _audit(user_id, run_id, agent, tool, args, status, result, latency_ms):
    try:
        await db.log_tool_call(user_id, agent, tool, args, status, result, latency_ms, run_id=run_id)
    except Exception:
        # auditing must never break the agent loop
        pass


async def _find_cached(run_id, tool, args):
    try:
        return await db.find_successful_tool_call(run_id, tool, args)
    except Exception:
        return None
