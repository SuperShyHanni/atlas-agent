import json
from typing import Dict
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from state import AcademicState

COORDINATOR_PROMPT = """You are ATLAS Coordinator - the orchestrator of an Academic Task and Learning Agent System.

You coordinate three specialized agents:
• PLANNER: Schedules, calendar management, deadlines, study sessions, time management
• NOTEWRITER: Study notes, summaries, study guides, flashcards, content creation
• ADVISOR: Academic strategies, learning tips, performance analysis, motivation, guidance

Analyze the student's request and decide which agents are needed.

Student: {student_name} ({major}, {year})
Learning Style: {learning_style}
Current Courses: {courses}
Upcoming Events: {upcoming_events}
Active Tasks: {active_tasks}

Request: {request}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "required_agents": ["PLANNER"],
  "priority_agent": "PLANNER",
  "reasoning": "brief explanation"
}}

Routing guidelines:
- PLANNER for: schedule, time, calendar, deadline, study plan, when to study, time management
- NOTEWRITER for: notes, summary, study guide, flashcard, explain content, create material
- ADVISOR for: advice, tips, strategy, how to improve, recommendation, guidance, motivation
- Use 1-3 agents depending on the request complexity"""


async def coordinator_node(state: AcademicState, llm, config: RunnableConfig = None) -> Dict:
    profile = state.get("profile", {})
    calendar = state.get("calendar", {})
    tasks = state.get("tasks", {})
    request = state["messages"][-1].content

    personal_info = profile.get("personal_info", {})
    courses = [c.get("name", "") for c in profile.get("academic_info", {}).get("current_courses", [])]
    events = calendar.get("events", [])
    task_list = tasks.get("tasks", [])

    prompt = COORDINATOR_PROMPT.format(
        student_name=personal_info.get("name", "Student"),
        major=personal_info.get("major", "Unknown"),
        year=personal_info.get("academic_year", "Unknown"),
        learning_style=profile.get("learning_preferences", {}).get("learning_style", "Unknown"),
        courses=", ".join(courses) if courses else "None listed",
        upcoming_events=len(events),
        active_tasks=len([t for t in task_list if not t.get("completed", False)]),
        request=request,
    )

    invoke_kwargs = {}
    if config:
        invoke_kwargs["config"] = config

    response = await llm.ainvoke([SystemMessage(content=prompt)], **invoke_kwargs)

    try:
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        analysis = json.loads(content[start:end]) if start >= 0 and end > start else {}
    except Exception:
        analysis = {}

    analysis.setdefault("required_agents", ["ADVISOR"])
    analysis.setdefault("priority_agent", analysis["required_agents"][0])
    analysis.setdefault("reasoning", "Default routing")

    return {
        "results": {"coordinator_analysis": analysis},
        "current_agent": "coordinator",
    }
