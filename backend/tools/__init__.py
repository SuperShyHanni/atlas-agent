from .calendar_tools import search_calendar, get_upcoming_deadlines
from .task_tools import get_pending_tasks, get_task_by_priority
from .profile_tools import get_learning_style, get_student_goals
from .notion_tools import notion_search, notion_get_page, notion_create_page
from .filesystem_tools import list_files, read_file

PLANNER_TOOLS = [search_calendar, get_upcoming_deadlines, get_pending_tasks, get_task_by_priority]
NOTEWRITER_TOOLS = [get_learning_style, list_files, read_file, notion_search, notion_get_page, notion_create_page]
ADVISOR_TOOLS = [get_learning_style, get_student_goals, get_task_by_priority, notion_search]

__all__ = [
    "search_calendar", "get_upcoming_deadlines",
    "get_pending_tasks", "get_task_by_priority",
    "get_learning_style", "get_student_goals",
    "notion_search", "notion_get_page", "notion_create_page",
    "list_files", "read_file",
    "PLANNER_TOOLS", "NOTEWRITER_TOOLS", "ADVISOR_TOOLS",
]
