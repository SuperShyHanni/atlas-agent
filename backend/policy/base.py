"""Policy abstraction for LearnBench — the model side of the agent/environment loop.

A policy turns an observation into a structured action. It never mutates
environment state. ``BasePolicy`` is the interface; ``APIPolicy`` (in
``api_policy.py``) is the OpenAI-compatible backend. ``LocalQwenPolicy`` will be
added in a later phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from learnbench.action import TOOL_SPECS


class PolicyError(Exception):
    """Raised when a policy fails to produce a usable action."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": "policy_error", "message": self.message, "details": self.details}


class BasePolicy(ABC):
    """Turn an observation into a single structured action (a ``tool``/``arguments`` dict)."""

    @abstractmethod
    async def act(self, observation: dict, tools: list[dict]) -> dict:
        raise NotImplementedError


_TOOL_DESCRIPTIONS = {
    "get_student_state": "Return the student's current mastery, remaining days and minutes.",
    "get_prerequisites": "List a topic's prerequisites and whether each is satisfied.",
    "study_topic": "Study a topic for a number of minutes, increasing its mastery.",
    "take_quiz": "Take a quiz on a topic; the score reflects current mastery.",
    "finish_day": "Advance to the next day; resets today's remaining minutes.",
}

_JSON_TYPES = {int: "integer", str: "string", float: "number", bool: "boolean"}


def tool_schema() -> list[dict]:
    """Serialize the LearnBench action space as a JSON-schema-style tool list."""
    tools = []
    for name, spec in TOOL_SPECS.items():
        properties = {}
        required = []
        for arg, arg_spec in spec.items():
            prop = {"type": _JSON_TYPES.get(arg_spec.typ, "string")}
            if arg_spec.typ is int and arg_spec.min_value is not None:
                prop["minimum"] = arg_spec.min_value
            if arg_spec.typ is int and arg_spec.max_value is not None:
                prop["maximum"] = arg_spec.max_value
            properties[arg] = prop
            if arg_spec.required:
                required.append(arg)
        tools.append(
            {
                "name": name,
                "description": _TOOL_DESCRIPTIONS.get(name, ""),
                "parameters": {"type": "object", "properties": properties, "required": required},
            }
        )
    return tools
