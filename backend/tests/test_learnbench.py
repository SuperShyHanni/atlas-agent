"""Unit tests for the LearnBench MVP (Phase 1).

All tests are offline and LLM-free. Run from the repo root with::

    python -m unittest discover -s backend/tests -t backend

or directly::

    python backend/tests/test_learnbench.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learnbench import (
    AgentAction,
    KnowledgeGraph,
    LearnEnv,
    RewardBreakdown,
    Task,
    Trajectory,
    TrajectoryStep,
    build_default_graph,
    parse_action,
)
from learnbench.constraints import ensure_time_budget, ensure_valid_topic
from learnbench.errors import ActionValidationError, ConstraintViolation
from learnbench.knowledge_graph import Topic
from learnbench.reward import estimate_tokens, reward_learning


class TestActionValidation(unittest.TestCase):
    def test_valid_actions(self):
        for action in [
            {"tool": "get_student_state", "arguments": {}},
            {"tool": "finish_day", "arguments": {}},
            {"tool": "get_prerequisites", "arguments": {"topic": "probability"}},
            {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}},
            {"tool": "take_quiz", "arguments": {"topic": "probability"}},
        ]:
            parsed = parse_action(action)
            self.assertEqual(parsed.tool, action["tool"])

    def test_unknown_tool(self):
        with self.assertRaises(ActionValidationError) as ctx:
            parse_action({"tool": "fly_to_moon", "arguments": {}})
        self.assertEqual(ctx.exception.code, "UNKNOWN_TOOL")

    def test_missing_required_argument(self):
        with self.assertRaises(ActionValidationError):
            parse_action({"tool": "study_topic", "arguments": {"topic": "probability"}})

    def test_invalid_type(self):
        with self.assertRaises(ActionValidationError):
            parse_action(
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": "40"}}
            )

    def test_negative_minutes(self):
        with self.assertRaises(ActionValidationError):
            parse_action({"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 0}})

    def test_unexpected_argument(self):
        with self.assertRaises(ActionValidationError):
            parse_action({"tool": "get_student_state", "arguments": {"nope": 1}})

    def test_agent_action_passthrough(self):
        action = AgentAction(tool="take_quiz", arguments={"topic": "probability"})
        self.assertEqual(parse_action(action), action)


class TestKnowledgeGraph(unittest.TestCase):
    def test_default_graph_size_and_acyclic(self):
        graph = build_default_graph()
        self.assertTrue(20 <= len(graph.topic_ids()) <= 50)
        self.assertTrue(graph.has_topic("transformer"))

    def test_prerequisites(self):
        graph = build_default_graph()
        self.assertIn("attention", graph.direct_prerequisites("transformer"))
        self.assertIn("deep_learning", graph.direct_prerequisites("transformer"))
        self.assertEqual(graph.direct_prerequisites("arithmetic"), [])

    def test_unmet_prerequisites(self):
        graph = build_default_graph()
        unmet = graph.unmet_prerequisites("transformer", {"attention": 0.9})
        self.assertIn("deep_learning", unmet)
        self.assertNotIn("attention", unmet)

    def test_unknown_prereq_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeGraph({"x": Topic("x", 0.5, 10, ("does_not_exist",))})

    def test_cycle_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeGraph(
                {
                    "a": Topic("a", 0.5, 10, ("b",)),
                    "b": Topic("b", 0.5, 10, ("a",)),
                }
            )


class TestConstraints(unittest.TestCase):
    def test_ensure_time_budget_raises(self):
        with self.assertRaises(ConstraintViolation):
            ensure_time_budget(50, 40)

    def test_ensure_valid_topic_raises(self):
        with self.assertRaises(ConstraintViolation):
            ensure_valid_topic("nope", build_default_graph())


class TestEnvBasics(unittest.TestCase):
    def test_reset_observation(self):
        env = LearnEnv()
        obs = env.reset(seed=42)
        self.assertEqual(obs["target_topic"], "transformer")
        self.assertEqual(obs["remaining_days"], 14)
        self.assertEqual(obs["remaining_minutes_today"], 180)
        self.assertIsNone(obs["last_quiz"])
        self.assertEqual(obs["mastery"]["transformer"], 0.0)

    def test_reset_seed_reproducibility(self):
        env1 = LearnEnv()
        env2 = LearnEnv()
        self.assertEqual(env1.reset(seed=42), env2.reset(seed=42))
        for _ in range(5):
            r1 = env1.step(
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}}
            )
            r2 = env2.step(
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}}
            )
            self.assertEqual(r1[0]["mastery"], r2[0]["mastery"])
            self.assertEqual(r1[1], r2[1])

    def test_env_step_shape(self):
        env = LearnEnv()
        env.reset(seed=7)
        obs, reward, done, info = env.step(
            {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}}
        )
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertIn("reward", info)
        self.assertIn("result", info)
        self.assertIn("mastery_before", info["result"])
        self.assertGreater(obs["mastery"]["probability"], 0.0)
        self.assertEqual(obs["remaining_minutes_today"], 180 - 40)


class TestStateAndTime(unittest.TestCase):
    def test_state_bounds(self):
        env = LearnEnv()
        env.reset(seed=1)
        for _ in range(50):
            env.step(
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}}
            )
        for value in env.state.mastery.values():
            self.assertTrue(0.0 <= value <= 1.0)

    def test_time_budget(self):
        env = LearnEnv()
        env.reset(seed=1)
        obs, reward, done, info = env.step(
            {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 181}}
        )
        self.assertFalse(done)
        self.assertIn("error", info)
        self.assertEqual(info["error"]["code"], "INSUFFICIENT_TIME")
        self.assertLess(reward, 0.0)
        self.assertEqual(env.state.remaining_minutes_today, 180)  # unchanged

    def test_time_never_negative(self):
        env = LearnEnv()
        env.reset(seed=2)
        for _ in range(100):
            env.step(
                {"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}}
            )
        self.assertGreaterEqual(env.state.remaining_minutes_today, 0)
        self.assertGreaterEqual(env.state.remaining_days, 0)

    def test_invalid_topic(self):
        env = LearnEnv()
        env.reset(seed=1)
        for action in [
            {"tool": "study_topic", "arguments": {"topic": "nope", "minutes": 10}},
            {"tool": "take_quiz", "arguments": {"topic": "nope"}},
            {"tool": "get_prerequisites", "arguments": {"topic": "nope"}},
        ]:
            obs, reward, done, info = env.step(action)
            self.assertFalse(done)
            self.assertIn("error", info)
            self.assertEqual(info["error"]["code"], "UNKNOWN_TOPIC")

    def test_prerequisite_violation(self):
        env = LearnEnv()
        env.reset(seed=3)
        obs, reward, done, info = env.step(
            {"tool": "study_topic", "arguments": {"topic": "transformer", "minutes": 40}}
        )
        self.assertFalse(done)
        self.assertTrue(info.get("prerequisite_violation"))
        self.assertFalse(info["result"]["prerequisites_met"])
        self.assertIn("attention", info["result"]["unmet_prerequisites"])
        self.assertLess(info["reward"]["prerequisite"], 0.0)


class TestQuiz(unittest.TestCase):
    def test_quiz_reproducibility(self):
        env1 = LearnEnv()
        env2 = LearnEnv()
        env1.reset(seed=99)
        env2.reset(seed=99)
        env1.step({"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}})
        env2.step({"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}})
        _, _, _, info1 = env1.step({"tool": "take_quiz", "arguments": {"topic": "probability"}})
        _, _, _, info2 = env2.step({"tool": "take_quiz", "arguments": {"topic": "probability"}})
        self.assertEqual(info1["result"]["score"], info2["result"]["score"])
        self.assertTrue(0 <= info1["result"]["score"] <= 100)


class TestTermination(unittest.TestCase):
    def test_target_reached(self):
        task = Task(
            id="arith",
            target_topic="arithmetic",
            target_mastery=0.1,
            total_days=3,
            minutes_per_day=200,
        )
        env = LearnEnv(task=task)
        env.reset(seed=1)
        obs, reward, done, info = env.step(
            {"tool": "study_topic", "arguments": {"topic": "arithmetic", "minutes": 60}}
        )
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "target_reached")
        self.assertGreater(info["reward"]["success"], 0.0)

    def test_out_of_time(self):
        task = Task(
            id="tight",
            target_topic="transformer",
            target_mastery=0.9,
            total_days=1,
            minutes_per_day=30,
        )
        env = LearnEnv(task=task)
        env.reset(seed=1)
        obs, reward, done, info = env.step({"tool": "finish_day", "arguments": {}})
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "out_of_time")
        self.assertEqual(info["reward"]["success"], 0.0)

    def test_max_steps(self):
        env = LearnEnv(max_steps=5)
        env.reset(seed=1)
        done = False
        info = {}
        for _ in range(5):
            _, _, done, info = env.step({"tool": "get_student_state", "arguments": {}})
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "max_steps")


class TestReward(unittest.TestCase):
    def test_components(self):
        self.assertEqual(reward_learning(-0.1), 0.0)
        self.assertAlmostEqual(reward_learning(0.3), 0.3)
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("hello world"), 0)

    def test_breakdown_total(self):
        breakdown = RewardBreakdown(
            success=1.0,
            learning=0.5,
            constraint=-1.0,
            prerequisite=-1.0,
            tool=1.0,
            call_cost=1.0,
            token_cost=100.0,
        )
        expected = (
            1.0 * 1.0
            + 0.5 * 1.0
            + (-1.0) * 1.0
            + (-1.0) * 0.5
            + 1.0 * 0.1
            - 1.0 * 0.01
            - 100.0 * 0.0001
        )
        self.assertAlmostEqual(breakdown.total, expected)


class TestTrajectorySerialization(unittest.TestCase):
    def test_roundtrip(self):
        traj = Trajectory(
            task_id="task_001",
            seed=42,
            steps=[
                TrajectoryStep(
                    observation={"mastery": {"probability": 0.3}},
                    action={
                        "tool": "study_topic",
                        "arguments": {"topic": "probability", "minutes": 40},
                    },
                    result={"gain": 0.2},
                    reward={"total": 0.2, "learning": 0.2},
                )
            ],
            final_reward=0.8,
            success=True,
            termination_reason="target_reached",
        )
        data = traj.to_dict()
        self.assertEqual(data["schema_version"], "1.0")
        restored = Trajectory.from_dict(data)
        self.assertEqual(restored, traj)
        self.assertEqual(restored.to_dict(), data)


class TestFuzzNoCrash(unittest.TestCase):
    def test_1000_random_steps(self):
        env = LearnEnv(max_steps=1000)
        env.reset(seed=123)
        topics = build_default_graph().topic_ids()
        for i in range(1000):
            choice = i % 8
            if choice == 0:
                action = {"tool": "get_student_state", "arguments": {}}
            elif choice == 1:
                action = {"tool": "get_prerequisites", "arguments": {"topic": topics[i % len(topics)]}}
            elif choice == 2:
                action = {
                    "tool": "study_topic",
                    "arguments": {"topic": topics[i % len(topics)], "minutes": (i % 60) + 1},
                }
            elif choice == 3:
                action = {"tool": "take_quiz", "arguments": {"topic": topics[i % len(topics)]}}
            elif choice == 4:
                action = {"tool": "finish_day", "arguments": {}}
            elif choice == 5:
                action = {"tool": "study_topic", "arguments": {"topic": "bad_topic", "minutes": 10}}
            elif choice == 6:
                action = {
                    "tool": "study_topic",
                    "arguments": {"topic": topics[i % len(topics)], "minutes": -5},
                }
            else:
                action = {"tool": "no_such_tool", "arguments": {}}
            # must never raise
            env.step(action)

        self.assertGreaterEqual(env.state.remaining_minutes_today, 0)
        self.assertGreaterEqual(env.state.remaining_days, 0)
        for value in env.state.mastery.values():
            self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
