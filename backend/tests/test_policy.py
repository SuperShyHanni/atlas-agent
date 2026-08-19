"""Unit tests for the LearnBench policy abstraction (Phase 2).

Offline: no LLM, no API key. ``_extract_action`` is tested directly; the live
``act()`` path is covered by an optional, skipped integration test.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy import APIPolicy, PolicyError, tool_schema


class TestToolSchema(unittest.TestCase):
    def test_five_tools(self):
        names = {t["name"] for t in tool_schema()}
        self.assertEqual(
            names,
            {"get_student_state", "get_prerequisites", "study_topic", "take_quiz", "finish_day"},
        )

    def test_study_topic_schema(self):
        tools = {t["name"]: t for t in tool_schema()}
        study = tools["study_topic"]
        self.assertEqual(study["parameters"]["properties"]["topic"]["type"], "string")
        minutes = study["parameters"]["properties"]["minutes"]
        self.assertEqual(minutes["type"], "integer")
        self.assertEqual(minutes["minimum"], 1)
        self.assertEqual(study["parameters"]["required"], ["topic", "minutes"])


class TestExtractAction(unittest.TestCase):
    def test_plain_json(self):
        out = APIPolicy._extract_action(
            '{"tool": "study_topic", "arguments": {"topic": "probability", "minutes": 40}}'
        )
        self.assertEqual(out["tool"], "study_topic")
        self.assertEqual(out["arguments"]["minutes"], 40)

    def test_markdown_fence(self):
        text = '```json\n{"tool": "take_quiz", "arguments": {"topic": "probability"}}\n```'
        out = APIPolicy._extract_action(text)
        self.assertEqual(out["tool"], "take_quiz")

    def test_surrounding_prose(self):
        text = 'Sure! Here is the action: {"tool": "finish_day", "arguments": {}} hope it helps.'
        out = APIPolicy._extract_action(text)
        self.assertEqual(out["tool"], "finish_day")

    def test_invalid_json_raises(self):
        with self.assertRaises(PolicyError):
            APIPolicy._extract_action('{"tool": "study_topic", ')

    def test_empty_raises(self):
        with self.assertRaises(PolicyError):
            APIPolicy._extract_action("")

    def test_non_object_raises(self):
        with self.assertRaises(PolicyError):
            APIPolicy._extract_action("[1, 2, 3]")


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set — skipping live API test")
class TestAPIPolicyLive(unittest.TestCase):
    def test_live_act_returns_action(self):
        try:
            policy = APIPolicy()
        except Exception as exc:  # noqa: BLE001 — deps may be absent in CI
            self.skipTest(f"cannot construct live policy (missing deps?): {exc}")
        import asyncio

        observation = {
            "mastery": {},
            "remaining_days": 14,
            "remaining_minutes_today": 180,
            "target_topic": "transformer",
            "target_mastery": 0.8,
        }
        action = asyncio.run(policy.act(observation, tool_schema()))
        self.assertIsInstance(action, dict)
        self.assertIn("tool", action)


if __name__ == "__main__":
    unittest.main()
