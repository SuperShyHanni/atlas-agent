"""Observation builder for LearnBench.

The observation exposes only what the agent is allowed to know — the student's
current mastery, time budget and quiz history. Hidden simulator parameters
(learning speed, noise, RNG state) are never included.
"""

from __future__ import annotations

from .state import StudentLearningState


def build_observation(state: StudentLearningState) -> dict:
    last_quiz = state.quiz_history[-1] if state.quiz_history else None
    return {
        "mastery": dict(state.mastery),
        "remaining_days": state.remaining_days,
        "remaining_minutes_today": state.remaining_minutes_today,
        "target_topic": state.target_topic,
        "target_mastery": state.target_mastery,
        "completed_topics": list(state.completed_topics),
        "last_quiz": last_quiz,
    }
