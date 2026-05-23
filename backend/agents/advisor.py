import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState
from tools import ADVISOR_TOOLS

ADVISOR_SYSTEM_PROMPT = """You are ATLAS Advisor - a specialized academic guidance and strategy agent.

Your role: provide personalized learning strategies, performance analysis, and motivation.

Coordinator Context: {coordinator_reasoning}

You have access to tools:
- get_learning_style: fetch learning preferences and study habits
- get_student_goals: fetch academic goals, grades, and challenges
- get_task_by_priority: see what high-priority work is pending

Instructions:
1. Use tools to ground your advice in the student's actual situation
2. Apply ReACT reasoning:
   - Thought: What does this student actually need based on their profile and goals?
   - Action: What strategies align with their learning style and challenges?
   - Advice: Concrete, personalized recommendations
3. Be empathetic and encouraging
4. Reference specific goals, grades, and challenges from the tool results
5. Give actionable steps, not generic advice"""


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
    for m in state.get("messages", []):
        if hasattr(m, "tool_calls") or m.__class__.__name__ == "ToolMessage":
            messages.append(m)
    messages.append(HumanMessage(content=request))
    return messages
