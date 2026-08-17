"""LearnBench episode state — deliberately separate from the product ``AcademicState``.

``StudentLearningState`` is the environment's internal source of truth for a
single learning-planning rollout. It is a plain dataclass with no LLM or
LangGraph dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A topic is considered "learned" (and eligible to satisfy a prerequisite) once
# its mastery reaches this bar.
MASTERED_THRESHOLD = 0.8


@dataclass
class StudentLearningState:
    mastery: dict[str, float]  # topic id -> mastery in [0, 1]
    remaining_days: int  # full days left until the deadline
    remaining_minutes_today: int  # study minutes left today
    target_topic: str  # topic the student must master
    target_mastery: float  # mastery required on target_topic
    minutes_per_day: int = 180  # minutes refreshed by finish_day
    learning_speed: float = 1.0  # hidden simulator parameter (not exposed to the model)
    quiz_history: list[dict] = field(default_factory=list)
    completed_topics: list[str] = field(default_factory=list)

    def recompute_completed(self) -> None:
        """Sync ``completed_topics`` with the current mastery dict."""
        self.completed_topics = [
            topic for topic, value in self.mastery.items() if value >= MASTERED_THRESHOLD
        ]
