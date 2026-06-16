from datetime import datetime, timezone
from typing import Optional
from langchain_core.tools import tool

# Per-request state is injected via an async-safe ContextVar (see tools/context.py).
from .context import get_state, set_state  # noqa: F401  (set_state re-exported for graph.py)


@tool
def search_calendar(keyword: str = "", days_ahead: int = 7) -> str:
    """
    Search upcoming calendar events within the next N days.
    Use this to find classes, exams, meetings, or deadlines.

    Args:
        keyword: Optional keyword to filter events by title or course name
        days_ahead: How many days ahead to search (default 7)
    """
    events = get_state().get("calendar", {}).get("events", [])
    if not events:
        return "No calendar events found."

    now = datetime.now(timezone.utc)
    results = []
    for e in events:
        try:
            start_str = e.get("start", {}).get("dateTime", "")
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            if not start_dt.tzinfo:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            delta = (start_dt - now).days
            if 0 <= delta <= days_ahead:
                title = e.get("title", "Event")
                if keyword.lower() in title.lower() or keyword.lower() in e.get("course", "").lower() or not keyword:
                    results.append(
                        f"• {title} — {start_str[:16]} (in {delta} days)"
                        + (f" [{e['course']}]" if e.get("course") else "")
                    )
        except Exception:
            continue

    return "\n".join(results) if results else f"No events found in the next {days_ahead} days."


@tool
def get_upcoming_deadlines(days_ahead: int = 7) -> str:
    """
    Get all upcoming task deadlines and exam dates sorted by urgency.
    Use this to understand what's most time-sensitive for the student.

    Args:
        days_ahead: How many days ahead to look (default 7)
    """
    state = get_state()
    tasks = state.get("tasks", {}).get("tasks", [])
    events = state.get("calendar", {}).get("events", [])

    now = datetime.now(timezone.utc).date()
    deadlines = []

    for t in tasks:
        if t.get("completed"):
            continue
        due = t.get("due_date", "")
        if due:
            try:
                due_date = datetime.fromisoformat(due).date()
                delta = (due_date - now).days
                if 0 <= delta <= days_ahead:
                    priority = t.get("priority", "medium").upper()
                    deadlines.append((delta, f"[TASK/{priority}] {t['title']} — due {due}"))
            except Exception:
                continue

    for e in events:
        title = e.get("title", "")
        if any(kw in title.lower() for kw in ["exam", "test", "quiz", "midterm", "final"]):
            try:
                start_str = e.get("start", {}).get("dateTime", "")
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if not start_dt.tzinfo:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                delta = (start_dt.date() - now).days
                if 0 <= delta <= days_ahead:
                    deadlines.append((delta, f"[EXAM] {title} — {start_str[:16]}"))
            except Exception:
                continue

    deadlines.sort(key=lambda x: x[0])
    lines = [d[1] for d in deadlines]
    return "\n".join(lines) if lines else f"No deadlines in the next {days_ahead} days."
