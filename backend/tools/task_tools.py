from langchain_core.tools import tool

_current_state = {}

def set_state(state: dict):
    global _current_state
    _current_state = state


@tool
def get_pending_tasks(course: str = "") -> str:
    """
    Get all incomplete tasks, optionally filtered by course name.
    Use this to understand what assignments or work the student has left.

    Args:
        course: Optional course name to filter tasks (e.g. 'Database Systems')
    """
    tasks = _current_state.get("tasks", {}).get("tasks", [])
    pending = [t for t in tasks if not t.get("completed", False)]

    if course:
        pending = [t for t in pending if course.lower() in t.get("course", "").lower()]

    if not pending:
        return f"No pending tasks{' for ' + course if course else ''}."

    lines = []
    for t in pending:
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority", "medium"), "⚪")
        lines.append(
            f"{priority_icon} {t['title']}"
            + (f" (due: {t['due_date']})" if t.get("due_date") else "")
            + (f" [{t['course']}]" if t.get("course") else "")
        )

    return "\n".join(lines)


@tool
def get_task_by_priority(priority: str) -> str:
    """
    Get pending tasks filtered by priority level.
    Use this to focus on what matters most.

    Args:
        priority: One of 'high', 'medium', or 'low'
    """
    tasks = _current_state.get("tasks", {}).get("tasks", [])
    filtered = [
        t for t in tasks
        if not t.get("completed") and t.get("priority", "medium").lower() == priority.lower()
    ]

    if not filtered:
        return f"No {priority}-priority tasks pending."

    lines = [
        f"• {t['title']}"
        + (f" — due {t['due_date']}" if t.get("due_date") else "")
        + (f" [{t.get('course', '')}]" if t.get("course") else "")
        for t in filtered
    ]
    return f"{priority.upper()} priority tasks:\n" + "\n".join(lines)
