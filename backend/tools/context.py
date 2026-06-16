"""Async-safe per-request state injection for tools.

Each chat request runs in its own asyncio task; a ContextVar isolates the
injected state per task, so concurrent users never read each other's profile,
calendar or tasks (fixes the cross-session leak of the old module-global state).
"""
import contextvars
from typing import Dict

_state_var: "contextvars.ContextVar[Dict]" = contextvars.ContextVar("agent_state", default={})


def set_state(state: Dict) -> None:
    _state_var.set(state or {})


def get_state() -> Dict:
    return _state_var.get()
