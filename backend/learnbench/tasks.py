"""Task definitions for LearnBench episodes.

A task pins the target topic, required mastery, and time budget. It is frozen
and independent of any LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    target_topic: str
    target_mastery: float
    total_days: int
    minutes_per_day: int
    learning_speed: float = 1.0
    initial_mastery: dict[str, float] = field(default_factory=dict)


def default_task() -> Task:
    """The canonical MVP task: master 'transformer' to 0.8 in 14 days."""
    return Task(
        id="transformer_14d",
        target_topic="transformer",
        target_mastery=0.8,
        total_days=14,
        minutes_per_day=180,
    )


def sample_tasks() -> list[Task]:
    """A few hand-picked tasks of varying difficulty for benchmarks."""
    return [
        default_task(),
        Task(
            id="probability_5d",
            target_topic="probability",
            target_mastery=0.8,
            total_days=5,
            minutes_per_day=120,
        ),
        Task(
            id="transformer_tight_7d",
            target_topic="transformer",
            target_mastery=0.9,
            total_days=7,
            minutes_per_day=180,
        ),
    ]
