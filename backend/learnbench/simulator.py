"""Rule-based student simulator for LearnBench (no LLM).

Learning gains are a transparent function of minutes studied, current mastery,
topic difficulty, prerequisite satisfaction, learning speed and small seeded
noise. Quiz scores track mastery with modest Gaussian noise. Both are
deterministic under a fixed ``random.Random`` seed.
"""

from __future__ import annotations

import random

from .knowledge_graph import Topic

LEARNING_RATE = 0.008  # mastery gained per minute at mastery 0, difficulty 0
DIFFICULTY_SCALE = 0.5  # difficulty 1 halves the base rate
PREREQ_MISS_PENALTY = 0.5  # studying without prerequisites halves the rate
NOISE_RANGE = (0.95, 1.05)  # multiplicative study noise
QUIZ_NOISE_SIGMA = 8.0  # quiz score noise (points, out of 100)


def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def apply_study(
    topic: Topic,
    mastery_before: float,
    minutes: int,
    learning_speed: float,
    prerequisites_met: bool,
    rng: random.Random,
) -> float:
    """Return the new (clamped) mastery after studying ``minutes`` on ``topic``."""
    difficulty_factor = 1.0 - DIFFICULTY_SCALE * topic.difficulty
    prereq_factor = 1.0 if prerequisites_met else PREREQ_MISS_PENALTY
    noise = rng.uniform(*NOISE_RANGE)
    raw_gain = (
        LEARNING_RATE
        * minutes
        * (1.0 - mastery_before)
        * difficulty_factor
        * prereq_factor
        * learning_speed
        * noise
    )
    return _clamp(mastery_before + raw_gain)


def quiz_score(mastery: float, rng: random.Random) -> int:
    """Return a 0–100 quiz score correlated with ``mastery`` plus Gaussian noise."""
    raw = mastery * 100.0 + rng.gauss(0.0, QUIZ_NOISE_SIGMA)
    return max(0, min(100, int(round(raw))))
