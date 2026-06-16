from langchain_core.tools import tool

# Per-request state is injected via an async-safe ContextVar (see tools/context.py).
from .context import get_state, set_state  # noqa: F401  (set_state re-exported for graph.py)


@tool
def get_learning_style() -> str:
    """
    Get the student's learning style and study preferences.
    Use this to tailor advice and study materials to the student.
    """
    profile = get_state().get("profile", {})
    prefs = profile.get("learning_preferences", {})
    personal = profile.get("personal_info", {})

    if not prefs:
        return "No learning preferences found in profile."

    return (
        f"Student: {personal.get('name', 'Unknown')} ({personal.get('major', 'Unknown')}, {personal.get('academic_year', 'Unknown')})\n"
        f"Learning style: {prefs.get('learning_style', 'Unknown')}\n"
        f"Peak study hours: {prefs.get('peak_study_hours', 'Unknown')}\n"
        f"Study environment: {prefs.get('study_environment', 'Unknown')}\n"
        f"Study hours/day: {prefs.get('study_hours_per_day', 'Unknown')}\n"
        f"Note-taking method: {prefs.get('note_taking_method', 'Unknown')}\n"
        f"Break interval: {prefs.get('break_interval_minutes', 'Unknown')} minutes"
    )


@tool
def get_student_goals() -> str:
    """
    Get the student's academic goals and current performance.
    Use this to align advice with what the student is trying to achieve.
    """
    profile = get_state().get("profile", {})
    goals = profile.get("goals", [])
    performance = profile.get("academic_info", {}).get("performance", {})
    challenges = profile.get("academic_info", {}).get("challenges", [])

    lines = []
    if goals:
        lines.append("Goals:\n" + "\n".join(f"  → {g}" for g in goals))
    if performance:
        lines.append("Current grades:\n" + "\n".join(f"  {k}: {v}" for k, v in performance.items()))
    if challenges:
        lines.append("Challenges:\n" + "\n".join(f"  ⚠ {c}" for c in challenges))

    return "\n\n".join(lines) if lines else "No goals or performance data found."
