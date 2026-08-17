"""Programmatic reward decomposition for LearnBench.

Reward components are pure, verifiable functions (no learned reward model), so
each can be unit-tested independently. ``RewardBreakdown`` combines them with
configurable weights and logs every component for later debugging::

    total = w_success*success + w_learning*learning + w_constraint*constraint
          + w_prereq*prerequisite + w_tool*tool - w_call*call_cost - w_token*token_cost
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    w_success: float = 1.0
    w_learning: float = 1.0
    w_constraint: float = 1.0
    w_prereq: float = 0.5
    w_tool: float = 0.1
    w_call: float = 0.01
    w_token: float = 0.0001


@dataclass
class RewardBreakdown:
    success: float = 0.0
    learning: float = 0.0
    constraint: float = 0.0
    prerequisite: float = 0.0
    tool: float = 0.0
    call_cost: float = 0.0
    token_cost: float = 0.0
    weights: RewardWeights = field(default_factory=RewardWeights)

    @property
    def total(self) -> float:
        return (
            self.weights.w_success * self.success
            + self.weights.w_learning * self.learning
            + self.weights.w_constraint * self.constraint
            + self.weights.w_prereq * self.prerequisite
            + self.weights.w_tool * self.tool
            - self.weights.w_call * self.call_cost
            - self.weights.w_token * self.token_cost
        )

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "success": self.success,
            "learning": self.learning,
            "constraint": self.constraint,
            "prerequisite": self.prerequisite,
            "tool": self.tool,
            "call_cost": self.call_cost,
            "token_cost": self.token_cost,
        }


def reward_success(target_reached: bool) -> float:
    return 1.0 if target_reached else 0.0


def reward_learning(mastery_gain: float) -> float:
    return max(0.0, mastery_gain)


def reward_constraint(num_violations: int) -> float:
    return -float(num_violations)


def reward_prerequisite(prerequisites_met: bool) -> float:
    return 0.0 if prerequisites_met else -1.0


def reward_tool(valid: bool) -> float:
    return 1.0 if valid else 0.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token); only used for the token-cost term."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
