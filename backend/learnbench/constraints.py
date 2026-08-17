"""Hard constraint guards for LearnBench.

These raise :class:`ConstraintViolation` with a stable code when a state or
action violates an invariant. The environment catches them and converts them
into a reward penalty, so violations are detected without crashing a rollout.
"""

from __future__ import annotations

from .errors import (
    DEADLINE_PASSED,
    INSUFFICIENT_TIME,
    INVALID_MASTERY,
    INVALID_TASK,
    UNKNOWN_TOPIC,
    ConstraintViolation,
)
from .knowledge_graph import KnowledgeGraph
from .tasks import Task


def ensure_mastery_in_bounds(mastery: dict[str, float]) -> None:
    for topic, value in mastery.items():
        if not 0.0 <= value <= 1.0:
            raise ConstraintViolation(INVALID_MASTERY, f"mastery for {topic!r} out of [0,1]: {value}")


def ensure_valid_topic(topic_id: str, graph: KnowledgeGraph) -> None:
    if not graph.has_topic(topic_id):
        raise ConstraintViolation(UNKNOWN_TOPIC, f"unknown topic {topic_id!r}")


def ensure_time_budget(minutes: int, remaining_minutes_today: int) -> None:
    if minutes <= 0:
        raise ConstraintViolation(INSUFFICIENT_TIME, f"minutes must be positive, got {minutes}")
    if minutes > remaining_minutes_today:
        raise ConstraintViolation(
            INSUFFICIENT_TIME,
            f"need {minutes} minutes but only {remaining_minutes_today} left today",
        )


def ensure_day_available(remaining_days: int) -> None:
    if remaining_days <= 0:
        raise ConstraintViolation(DEADLINE_PASSED, "no days remaining before the deadline")


def ensure_valid_task(task: Task, graph: KnowledgeGraph) -> None:
    if not graph.has_topic(task.target_topic):
        raise ConstraintViolation(INVALID_TASK, f"task target topic {task.target_topic!r} is unknown")
    if not 0.0 < task.target_mastery <= 1.0:
        raise ConstraintViolation(
            INVALID_TASK, f"target_mastery must be in (0, 1], got {task.target_mastery}"
        )
    if task.total_days < 1:
        raise ConstraintViolation(INVALID_TASK, f"total_days must be >= 1, got {task.total_days}")
    if task.minutes_per_day < 1:
        raise ConstraintViolation(
            INVALID_TASK, f"minutes_per_day must be >= 1, got {task.minutes_per_day}"
        )
    for topic, value in task.initial_mastery.items():
        if not graph.has_topic(topic):
            raise ConstraintViolation(INVALID_TASK, f"initial_mastery references unknown topic {topic!r}")
        if not 0.0 <= value <= 1.0:
            raise ConstraintViolation(INVALID_TASK, f"initial mastery for {topic!r} out of [0,1]: {value}")
