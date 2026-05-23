from .calendar_tools import search_calendar, get_upcoming_deadlines
from .task_tools import get_pending_tasks, get_task_by_priority
from .profile_tools import get_learning_style, get_student_goals

PLANNER_TOOLS = [search_calendar, get_upcoming_deadlines, get_pending_tasks, get_task_by_priority]
NOTEWRITER_TOOLS = [get_learning_style]
ADVISOR_TOOLS = [get_learning_style, get_student_goals, get_task_by_priority]

__all__ = [
    "search_calendar", "get_upcoming_deadlines",
    "get_pending_tasks", "get_task_by_priority",
    "get_learning_style", "get_student_goals",
    "PLANNER_TOOLS", "NOTEWRITER_TOOLS", "ADVISOR_TOOLS",
]
