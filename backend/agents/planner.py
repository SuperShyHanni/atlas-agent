import json
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState

PLANNER_SYSTEM_PROMPT = """You are ATLAS Planner - a specialized academic scheduling and time management agent.

Your role is to help students:
- Create optimized study schedules
- Manage deadlines and calendar events
- Plan study sessions around existing commitments
- Prioritize tasks based on deadlines and importance

Student Profile:
{profile_summary}

Current Calendar Events:
{calendar_events}

Current Tasks & Deadlines:
{tasks_summary}

Coordinator Context:
{coordinator_reasoning}

Instructions:
1. Analyze the student's request in context of their schedule
2. Use ReACT reasoning:
   - Thought: What scheduling challenges exist?
   - Action: How to optimize the schedule?
   - Observation: What conflicts or opportunities exist?
   - Plan: Concrete schedule recommendation
3. Be specific with times, dates, and priorities
4. Consider the student's learning style and peak hours
5. Format your response in clear markdown with sections"""


async def planner_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    profile = state.get("profile", {})
    calendar = state.get("calendar", {})
    tasks = state.get("tasks", {})
    request = state["messages"][-1].content
    coordinator_analysis = state.get("results", {}).get("coordinator_analysis", {})

    personal_info = profile.get("personal_info", {})
    prefs = profile.get("learning_preferences", {})
    courses = profile.get("academic_info", {}).get("current_courses", [])

    profile_summary = {
        "name": personal_info.get("name", "Student"),
        "major": personal_info.get("major", "Unknown"),
        "year": personal_info.get("academic_year", "Unknown"),
        "learning_style": prefs.get("learning_style", "Unknown"),
        "study_hours_per_day": prefs.get("study_hours_per_day", "Unknown"),
        "peak_hours": prefs.get("peak_study_hours", "Unknown"),
        "courses": [{"name": c.get("name"), "credits": c.get("credits")} for c in courses],
    }

    events = calendar.get("events", [])
    calendar_text = "\n".join(
        [f"- {e.get('title', 'Event')} on {e.get('start', {}).get('dateTime', 'TBD')}" for e in events[:10]]
    ) or "No upcoming events"

    task_list = tasks.get("tasks", [])
    tasks_text = "\n".join(
        [
            f"- [{t.get('priority', 'medium').upper()}] {t.get('title', 'Task')} - Due: {t.get('due_date', 'No date')} {'✓' if t.get('completed') else '○'}"
            for t in task_list[:15]
        ]
    ) or "No tasks"

    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        profile_summary=json.dumps(profile_summary, indent=2),
        calendar_events=calendar_text,
        tasks_summary=tasks_text,
        coordinator_reasoning=coordinator_analysis.get("reasoning", ""),
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=request)]
    invoke_kwargs = {}
    if config:
        invoke_kwargs["config"] = config

    response = await llm.ainvoke(messages, **invoke_kwargs)

    return {
        "results": {"planner_response": response.content},
        "current_agent": "planner",
    }
