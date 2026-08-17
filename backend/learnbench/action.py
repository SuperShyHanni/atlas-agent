"""Structured actions for the learning-planning policy.

The MVP action space is intentionally tiny — five tools. ``AgentAction`` is a
validated ``(tool, arguments)`` pair; ``parse_action`` coerces and validates
raw dicts / existing actions and raises :class:`ActionValidationError` with a
stable code instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ActionValidationError, INVALID_ACTION, INVALID_ARGUMENTS, UNKNOWN_TOOL


@dataclass(frozen=True)
class ArgumentSpec:
    typ: type
    required: bool = True
    min_value: int | None = None
    max_value: int | None = None


TOOL_SPECS: dict[str, dict[str, ArgumentSpec]] = {
    "get_student_state": {},
    "get_prerequisites": {"topic": ArgumentSpec(str)},
    "study_topic": {"topic": ArgumentSpec(str), "minutes": ArgumentSpec(int, min_value=1)},
    "take_quiz": {"topic": ArgumentSpec(str)},
    "finish_day": {},
}

TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SPECS.keys())


@dataclass(frozen=True)
class AgentAction:
    tool: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": dict(self.arguments)}


def _check_type(value: Any, typ: type) -> bool:
    if typ is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if typ is str:
        return isinstance(value, str) and bool(value.strip())
    return isinstance(value, typ)


def _validate(tool: Any, arguments: Any) -> AgentAction:
    if not isinstance(tool, str) or tool not in TOOL_SPECS:
        known = ", ".join(TOOL_NAMES)
        raise ActionValidationError(UNKNOWN_TOOL, f"unknown tool {tool!r}; expected one of: {known}")
    if not isinstance(arguments, dict):
        raise ActionValidationError(INVALID_ARGUMENTS, "arguments must be an object")

    spec = TOOL_SPECS[tool]
    cleaned: dict[str, Any] = {}
    for name, arg_spec in spec.items():
        if name not in arguments:
            if arg_spec.required:
                raise ActionValidationError(
                    INVALID_ARGUMENTS, f"missing required argument {name!r} for tool {tool!r}"
                )
            continue
        value = arguments[name]
        if not _check_type(value, arg_spec.typ):
            raise ActionValidationError(
                INVALID_ARGUMENTS,
                f"argument {name!r} for tool {tool!r} must be {arg_spec.typ.__name__}",
            )
        if arg_spec.typ is int and arg_spec.min_value is not None and value < arg_spec.min_value:
            raise ActionValidationError(
                INVALID_ARGUMENTS,
                f"argument {name!r} for tool {tool!r} must be >= {arg_spec.min_value}",
            )
        if arg_spec.typ is int and arg_spec.max_value is not None and value > arg_spec.max_value:
            raise ActionValidationError(
                INVALID_ARGUMENTS,
                f"argument {name!r} for tool {tool!r} must be <= {arg_spec.max_value}",
            )
        cleaned[name] = value

    extra = set(arguments) - set(spec)
    if extra:
        raise ActionValidationError(
            INVALID_ARGUMENTS, f"unexpected argument(s) {sorted(extra)} for tool {tool!r}"
        )
    return AgentAction(tool=tool, arguments=cleaned)


def parse_action(action: Any) -> AgentAction:
    """Coerce ``action`` (dict or ``AgentAction``) into a validated ``AgentAction``."""
    if isinstance(action, AgentAction):
        tool, arguments = action.tool, action.arguments
    elif isinstance(action, dict):
        tool = action.get("tool")
        arguments = action.get("arguments", {})
    else:
        raise ActionValidationError(
            INVALID_ACTION, f"action must be a dict or AgentAction, got {type(action).__name__}"
        )
    return _validate(tool, arguments)
