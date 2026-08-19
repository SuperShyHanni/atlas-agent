"""Rollout runner for LearnBench — drives observation → action → env.step.

The runner is policy-independent: it takes any ``BasePolicy`` and a ``LearnEnv``
and records every step into a versioned ``Trajectory``. It enforces loop limits
and guards against runaway invalid-action or error loops. Invalid model output
never crashes a rollout; it is recorded and counted, and if the limit is hit the
rollout terminates with a structured reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from learnbench.action import parse_action
from learnbench.env import LearnEnv
from learnbench.errors import LearnBenchError
from learnbench.tasks import Task
from learnbench.trajectory import Trajectory, TrajectoryStep
from policy.base import BasePolicy, tool_schema


@dataclass
class RolloutConfig:
    max_steps: int = 50  # max policy→env cycles in one rollout
    max_invalid_actions: int = 10  # max invalid/error actions before giving up


@dataclass
class RolloutResult:
    trajectory: Trajectory
    success: bool
    termination_reason: str
    total_steps: int
    valid_actions: int
    invalid_actions: int

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "termination_reason": self.termination_reason,
            "total_steps": self.total_steps,
            "valid_actions": self.valid_actions,
            "invalid_actions": self.invalid_actions,
            "final_reward": self.trajectory.final_reward,
            "trajectory": self.trajectory.to_dict(),
        }


class RolloutRunner:
    def __init__(self, env: LearnEnv, policy: BasePolicy, config: RolloutConfig | None = None):
        self.env = env
        self.policy = policy
        self.config = config or RolloutConfig()

    async def run(self, task: Task | None = None, seed: int | None = None) -> RolloutResult:
        observation = self.env.reset(task=task, seed=seed)
        trajectory = Trajectory(task_id=self.env.task.id, seed=self.env.seed)
        tools = tool_schema()
        valid_actions = 0
        invalid_actions = 0
        steps = 0

        while steps < self.config.max_steps:
            steps += 1

            try:
                raw_action = await self.policy.act(observation, tools)
            except Exception as exc:  # noqa: BLE001 — a broken policy must not crash a rollout
                trajectory.steps.append(
                    TrajectoryStep(observation=observation, error=_policy_error(exc))
                )
                return self._finish(trajectory, steps, valid_actions, invalid_actions, "policy_error")

            try:
                action = parse_action(raw_action)
            except LearnBenchError as err:
                invalid_actions += 1
                trajectory.steps.append(
                    TrajectoryStep(
                        observation=observation,
                        action=_coerce_action(raw_action),
                        error=err.to_dict(),
                    )
                )
                if invalid_actions >= self.config.max_invalid_actions:
                    return self._finish(
                        trajectory, steps, valid_actions, invalid_actions, "too_many_invalid_actions"
                    )
                continue

            pre_observation = observation
            observation, _reward, done, info = self.env.step(action)
            valid_actions += 1
            trajectory.steps.append(
                TrajectoryStep(
                    observation=pre_observation,
                    action=action.to_dict(),
                    result=info.get("result", {}),
                    reward=info.get("reward", {}),
                )
            )
            if done:
                reason = info.get("termination_reason", "terminated")
                return self._finish(trajectory, steps, valid_actions, invalid_actions, reason)

        return self._finish(trajectory, steps, valid_actions, invalid_actions, "max_rollout_steps")

    @staticmethod
    def _finish(
        trajectory: Trajectory,
        total_steps: int,
        valid_actions: int,
        invalid_actions: int,
        reason: str,
    ) -> RolloutResult:
        final_reward = sum(
            step.reward.get("total", 0.0)
            for step in trajectory.steps
            if isinstance(step.reward, dict)
        )
        trajectory.final_reward = final_reward
        trajectory.success = reason == "target_reached"
        trajectory.termination_reason = reason
        return RolloutResult(
            trajectory=trajectory,
            success=trajectory.success,
            termination_reason=reason,
            total_steps=total_steps,
            valid_actions=valid_actions,
            invalid_actions=invalid_actions,
        )


def _coerce_action(action) -> dict:
    if isinstance(action, dict):
        return action
    return {"raw": str(action)}


def _policy_error(exc: Exception) -> dict:
    return {"code": "policy_error", "message": str(exc)}
