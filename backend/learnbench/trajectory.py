"""Versioned trajectory serialization for LearnBench rollouts.

Every rollout must be recordable, not just its final answer. ``Trajectory`` is
a versioned, JSON-serializable structure; later phases (recorder, SFT dataset,
GRPO) consume it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1.0"


@dataclass
class TrajectoryStep:
    observation: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str = ""
    seed: int = 0
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_reward: float = 0.0
    success: bool = False
    termination_reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "seed": self.seed,
            "steps": [asdict(step) for step in self.steps],
            "final_reward": self.final_reward,
            "success": self.success,
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        steps = [
            TrajectoryStep(
                observation=step.get("observation", {}),
                action=step.get("action", {}),
                result=step.get("result", {}),
                reward=step.get("reward", {}),
            )
            for step in data.get("steps", [])
        ]
        return cls(
            task_id=data.get("task_id", ""),
            seed=data.get("seed", 0),
            steps=steps,
            final_reward=data.get("final_reward", 0.0),
            success=data.get("success", False),
            termination_reason=data.get("termination_reason", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )
