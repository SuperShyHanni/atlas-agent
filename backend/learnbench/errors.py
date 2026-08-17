"""Structured error types for the LearnBench environment.

The environment must never crash on a bad action. Every recoverable failure
mode raises a :class:`LearnBenchError` carrying a stable machine-readable
``code`` plus a human message; ``LearnEnv.step`` catches it and converts it
into a reward penalty and an ``info`` entry.
"""

from __future__ import annotations

# Stable error codes. Keep these additive — rollouts / tests may match on them.
INVALID_ACTION = "INVALID_ACTION"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
UNKNOWN_TOPIC = "UNKNOWN_TOPIC"
INSUFFICIENT_TIME = "INSUFFICIENT_TIME"
DEADLINE_PASSED = "DEADLINE_PASSED"
PREREQUISITE_NOT_MET = "PREREQUISITE_NOT_MET"
INVALID_MASTERY = "INVALID_MASTERY"
INVALID_TASK = "INVALID_TASK"


class LearnBenchError(Exception):
    """Base error carrying a stable ``code`` and optional structured ``details``."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ActionValidationError(LearnBenchError):
    """Raised when an action fails tool/argument validation."""


class ConstraintViolation(LearnBenchError):
    """Raised when an otherwise-valid action violates an environment constraint."""
