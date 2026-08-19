"""Unit tests for the LearnBench rollout runner + trajectory recording (Phase 2).

Policy-independent: uses scripted/fake policies, no LLM, no API key.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learnbench import LearnEnv, Task, Trajectory
from policy.base import BasePolicy
from rollout import RolloutConfig, RolloutRunner


class ScriptedPolicy(BasePolicy):
    """Returns a fixed sequence of actions, repeating the last one."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.calls = 0

    async def act(self, observation, tools):
        action = self._actions[min(self.calls, len(self._actions) - 1)]
        self.calls += 1
        return action


class ErrorPolicy(BasePolicy):
    async def act(self, observation, tools):
        raise RuntimeError("boom")


class TestRollout(unittest.IsolatedAsyncioTestCase):
    async def test_reaches_target(self):
        task = Task(
            id="arith", target_topic="arithmetic", target_mastery=0.1,
            total_days=3, minutes_per_day=200,
        )
        env = LearnEnv(task=task)
        policy = ScriptedPolicy(
            [{"tool": "study_topic", "arguments": {"topic": "arithmetic", "minutes": 60}}]
        )
        result = await RolloutRunner(env, policy).run(seed=42)
        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "target_reached")
        self.assertGreater(result.trajectory.final_reward, 0.0)

    async def test_out_of_time(self):
        task = Task(
            id="tight", target_topic="transformer", target_mastery=0.9,
            total_days=1, minutes_per_day=30,
        )
        env = LearnEnv(task=task)
        policy = ScriptedPolicy([{"tool": "finish_day", "arguments": {}}])
        result = await RolloutRunner(env, policy).run(seed=1)
        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "out_of_time")

    async def test_max_steps(self):
        env = LearnEnv()
        policy = ScriptedPolicy([{"tool": "get_student_state", "arguments": {}}])
        result = await RolloutRunner(env, policy, RolloutConfig(max_steps=5)).run(seed=1)
        self.assertEqual(result.termination_reason, "max_rollout_steps")
        self.assertEqual(result.total_steps, 5)
        self.assertEqual(result.valid_actions, 5)

    async def test_too_many_invalid_actions(self):
        env = LearnEnv()
        policy = ScriptedPolicy([{"tool": "not_a_tool", "arguments": {}}])
        result = await RolloutRunner(
            env, policy, RolloutConfig(max_invalid_actions=3, max_steps=100)
        ).run(seed=1)
        self.assertEqual(result.termination_reason, "too_many_invalid_actions")
        self.assertEqual(result.invalid_actions, 3)
        self.assertFalse(result.success)

    async def test_invalid_then_valid(self):
        env = LearnEnv()
        policy = ScriptedPolicy(
            [
                {"tool": "study_topic", "arguments": {"topic": "probability"}},  # missing minutes
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}},
                {"tool": "get_student_state", "arguments": {}},
            ]
        )
        result = await RolloutRunner(env, policy, RolloutConfig(max_steps=3)).run(seed=1)
        self.assertEqual(result.invalid_actions, 1)
        self.assertEqual(result.valid_actions, 2)
        self.assertIsNotNone(result.trajectory.steps[0].error)
        self.assertIsNone(result.trajectory.steps[1].error)
        self.assertIn("total", result.trajectory.steps[1].reward)

    async def test_policy_error(self):
        env = LearnEnv()
        result = await RolloutRunner(env, ErrorPolicy()).run(seed=1)
        self.assertEqual(result.termination_reason, "policy_error")
        self.assertEqual(result.valid_actions, 0)
        self.assertEqual(result.invalid_actions, 0)
        self.assertEqual(result.trajectory.steps[0].error["code"], "policy_error")

    async def test_trajectory_json_roundtrip(self):
        env = LearnEnv()
        policy = ScriptedPolicy(
            [
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}},
                {"tool": "take_quiz", "arguments": {"topic": "probability"}},
            ]
        )
        result = await RolloutRunner(env, policy, RolloutConfig(max_steps=2)).run(seed=42)
        data = result.trajectory.to_dict()
        self.assertEqual(data["task_id"], "transformer_14d")
        self.assertEqual(data["seed"], 42)
        restored = Trajectory.from_dict(data)
        self.assertEqual(restored, result.trajectory)
        self.assertEqual(json.loads(json.dumps(data)), data)

    async def test_final_reward_is_sum(self):
        env = LearnEnv()
        policy = ScriptedPolicy(
            [
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}},
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}},
            ]
        )
        result = await RolloutRunner(env, policy, RolloutConfig(max_steps=2)).run(seed=7)
        step_sum = sum(s.reward.get("total", 0.0) for s in result.trajectory.steps)
        self.assertAlmostEqual(result.trajectory.final_reward, step_sum)


if __name__ == "__main__":
    unittest.main()
