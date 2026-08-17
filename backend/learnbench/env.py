"""LearnBench environment — a Gym-like interface with no Gym dependency.

``LearnEnv`` runs a complete learning-planning episode from hand-written (or
later, policy-produced) actions, with no LLM required. It is the environment of
truth: the policy never mutates state directly.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .action import AgentAction, parse_action
from .constraints import (
    ensure_day_available,
    ensure_mastery_in_bounds,
    ensure_time_budget,
    ensure_valid_task,
    ensure_valid_topic,
)
from .errors import LearnBenchError
from .knowledge_graph import KnowledgeGraph, build_default_graph
from .observation import build_observation
from .reward import (
    RewardBreakdown,
    RewardWeights,
    estimate_tokens,
    reward_constraint,
    reward_learning,
    reward_prerequisite,
    reward_success,
    reward_tool,
)
from .simulator import apply_study, quiz_score
from .state import StudentLearningState
from .tasks import Task, default_task

DEFAULT_MAX_STEPS = 200


def _new_seed() -> int:
    return random.SystemRandom().randint(0, 2**31 - 1)


class LearnEnv:
    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        task: Task | None = None,
        weights: RewardWeights | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.graph = graph or build_default_graph()
        self.task = task or default_task()
        self.weights = weights or RewardWeights()
        self.max_steps = max_steps
        self._rng = random.Random()
        self.seed: int | None = None
        self.state: StudentLearningState | None = None
        self.step_count = 0
        self.termination_reason: str | None = None
        self.last_reward: RewardBreakdown | None = None

    # -- lifecycle ----------------------------------------------------------
    def reset(self, task: Task | None = None, seed: int | None = None) -> dict:
        if task is not None:
            self.task = task
        self.seed = seed if seed is not None else _new_seed()
        self._rng = random.Random(self.seed)

        ensure_valid_task(self.task, self.graph)
        initial = self.task.initial_mastery
        mastery = {topic_id: initial.get(topic_id, 0.0) for topic_id in self.graph.topic_ids()}
        self.state = StudentLearningState(
            mastery=mastery,
            remaining_days=self.task.total_days,
            remaining_minutes_today=self.task.minutes_per_day,
            target_topic=self.task.target_topic,
            target_mastery=self.task.target_mastery,
            minutes_per_day=self.task.minutes_per_day,
            learning_speed=self.task.learning_speed,
        )
        self.state.recompute_completed()
        ensure_mastery_in_bounds(self.state.mastery)
        self.step_count = 0
        self.termination_reason = None
        self.last_reward = None
        return build_observation(self.state)

    def step(self, action: Any) -> tuple[dict, float, bool, dict]:
        """Apply one action; return ``(observation, reward, done, info)``."""
        if self.state is None:
            raise RuntimeError("LearnEnv.step() called before reset()")

        self.step_count += 1

        try:
            parsed = parse_action(action)
        except LearnBenchError as err:
            return self._penalty_step(err, action, tool_valid=False)

        tool = parsed.tool
        if tool == "get_student_state":
            return self._finish_step(parsed, result={"student_state": build_observation(self.state)})
        if tool == "get_prerequisites":
            return self._prerequisites_step(parsed)
        if tool == "study_topic":
            return self._study_step(parsed)
        if tool == "take_quiz":
            return self._quiz_step(parsed)
        if tool == "finish_day":
            return self._finish_day_step(parsed)
        # unreachable: parse_action validates the tool name
        raise AssertionError(f"unhandled tool {tool!r}")

    # -- tool handlers ------------------------------------------------------
    def _prerequisites_step(self, parsed: AgentAction) -> tuple[dict, float, bool, dict]:
        topic = parsed.arguments["topic"]
        try:
            ensure_valid_topic(topic, self.graph)
        except LearnBenchError as err:
            return self._penalty_step(err, parsed, tool_valid=True)

        topic_obj = self.graph.get(topic)
        unmet = self.graph.unmet_prerequisites(topic, self.state.mastery)
        result = {
            "topic": topic,
            "prerequisites": list(topic_obj.prerequisites),
            "difficulty": topic_obj.difficulty,
            "study_minutes": topic_obj.study_minutes,
            "mastery": self.state.mastery.get(topic, 0.0),
            "prerequisites_met": not unmet,
            "unmet_prerequisites": unmet,
        }
        return self._finish_step(parsed, result=result)

    def _study_step(self, parsed: AgentAction) -> tuple[dict, float, bool, dict]:
        topic = parsed.arguments["topic"]
        minutes = parsed.arguments["minutes"]
        try:
            ensure_valid_topic(topic, self.graph)
            ensure_time_budget(minutes, self.state.remaining_minutes_today)
            ensure_day_available(self.state.remaining_days)
        except LearnBenchError as err:
            return self._penalty_step(err, parsed, tool_valid=True)

        topic_obj = self.graph.get(topic)
        prerequisites_met = self.graph.prerequisites_satisfied(topic, self.state.mastery)
        unmet = self.graph.unmet_prerequisites(topic, self.state.mastery)
        before = self.state.mastery.get(topic, 0.0)
        after = apply_study(
            topic_obj, before, minutes, self.state.learning_speed, prerequisites_met, self._rng
        )
        gain = after - before

        self.state.mastery[topic] = after
        self.state.remaining_minutes_today -= minutes
        self.state.recompute_completed()

        result = {
            "topic": topic,
            "minutes": minutes,
            "mastery_before": before,
            "mastery_after": after,
            "gain": gain,
            "prerequisites_met": prerequisites_met,
            "unmet_prerequisites": unmet,
        }
        return self._finish_step(
            parsed,
            result=result,
            learning=reward_learning(gain),
            prerequisite=reward_prerequisite(prerequisites_met),
            extra={"prerequisite_violation": not prerequisites_met},
        )

    def _quiz_step(self, parsed: AgentAction) -> tuple[dict, float, bool, dict]:
        topic = parsed.arguments["topic"]
        try:
            ensure_valid_topic(topic, self.graph)
            ensure_day_available(self.state.remaining_days)
        except LearnBenchError as err:
            return self._penalty_step(err, parsed, tool_valid=True)

        mastery = self.state.mastery.get(topic, 0.0)
        score = quiz_score(mastery, self._rng)
        self.state.quiz_history.append(
            {"topic": topic, "score": score, "mastery": mastery, "step": self.step_count}
        )
        result = {"topic": topic, "score": score, "mastery": mastery}
        return self._finish_step(parsed, result=result)

    def _finish_day_step(self, parsed: AgentAction) -> tuple[dict, float, bool, dict]:
        try:
            ensure_day_available(self.state.remaining_days)
        except LearnBenchError as err:
            return self._penalty_step(err, parsed, tool_valid=True)

        self.state.remaining_days -= 1
        self.state.remaining_minutes_today = self.state.minutes_per_day
        result = {
            "remaining_days": self.state.remaining_days,
            "remaining_minutes_today": self.state.remaining_minutes_today,
        }
        return self._finish_step(parsed, result=result)

    # -- helpers ------------------------------------------------------------
    def _finish_step(
        self,
        parsed: AgentAction,
        *,
        result: dict | None = None,
        learning: float = 0.0,
        prerequisite: float = 0.0,
        extra: dict | None = None,
    ) -> tuple[dict, float, bool, dict]:
        done, reason, success = self._termination()
        tokens = estimate_tokens(json.dumps(parsed.to_dict()))
        breakdown = RewardBreakdown(
            success=success,
            learning=learning,
            constraint=0.0,
            prerequisite=prerequisite,
            tool=reward_tool(True),
            call_cost=1.0,
            token_cost=float(tokens),
            weights=self.weights,
        )
        self.last_reward = breakdown
        info: dict[str, Any] = {"reward": breakdown.to_dict(), "step": self.step_count}
        if result is not None:
            info["result"] = result
        if extra:
            info.update(extra)
        if done:
            info["termination_reason"] = reason
            self.termination_reason = reason
        return build_observation(self.state), breakdown.total, done, info

    def _penalty_step(
        self, err: LearnBenchError, action: Any, tool_valid: bool
    ) -> tuple[dict, float, bool, dict]:
        if isinstance(action, AgentAction):
            action_dict = action.to_dict()
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {}
        tokens = estimate_tokens(json.dumps(action_dict))
        breakdown = RewardBreakdown(
            success=0.0,
            learning=0.0,
            constraint=reward_constraint(1),
            prerequisite=0.0,
            tool=reward_tool(tool_valid),
            call_cost=1.0,
            token_cost=float(tokens),
            weights=self.weights,
        )
        self.last_reward = breakdown
        info = {
            "error": err.to_dict(),
            "valid_action": tool_valid,
            "reward": breakdown.to_dict(),
            "step": self.step_count,
        }
        return build_observation(self.state), breakdown.total, False, info

    def _termination(self) -> tuple[bool, str | None, float]:
        if self.state.mastery.get(self.state.target_topic, 0.0) >= self.state.target_mastery:
            return True, "target_reached", reward_success(True)
        if self.state.remaining_days <= 0:
            return True, "out_of_time", reward_success(False)
        if self.step_count >= self.max_steps:
            return True, "max_steps", reward_success(False)
        return False, None, 0.0
