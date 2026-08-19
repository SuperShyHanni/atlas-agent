"""OpenAI-compatible API policy for LearnBench.

Reuses the production LLM path (``graph.get_llm``) lazily so that importing
this module never pulls in LangGraph / the app graph, and the offline test
suite stays dependency-free. The LLM is asked for a single JSON action; output
is parsed defensively into an action dict.
"""

from __future__ import annotations

import json

from .base import BasePolicy, PolicyError

DEFAULT_INSTRUCTIONS = (
    "You control a student's study plan in a learning environment.\n\n"
    "Respond with a SINGLE JSON action and nothing else — no prose, no markdown fences.\n"
    'The action must look like: {"tool": "<tool_name>", "arguments": {...}}.\n\n'
    "Tool reference:\n"
    "- get_student_state (no args): view mastery / remaining time.\n"
    '- get_prerequisites {"topic": "<id>"}: list a topic\'s prerequisites.\n'
    '- study_topic {"topic": "<id>", "minutes": <int>}: study to raise mastery.\n'
    '- take_quiz {"topic": "<id>"}: check current mastery via a quiz.\n'
    "- finish_day (no args): advance to the next day (resets daily minutes).\n\n"
    "Available tools (JSON schema):"
)


def default_llm():
    """Return the production OpenAI-compatible ``ChatOpenAI`` (reuses graph.get_llm)."""
    from graph import get_llm  # lazy: graph.py builds the whole app on import

    return get_llm()


class APIPolicy(BasePolicy):
    def __init__(self, llm=None, instructions: str | None = None):
        self._llm = llm  # injectable for tests; default_llm() is used lazily when None
        self._instructions = instructions or DEFAULT_INSTRUCTIONS

    async def act(self, observation: dict, tools: list[dict]) -> dict:
        llm = self._llm if self._llm is not None else default_llm()

        from langchain_core.messages import HumanMessage, SystemMessage  # lazy import

        system = SystemMessage(content=self._render_system(tools))
        user = HumanMessage(content=json.dumps(observation, ensure_ascii=False, indent=2))
        response = await llm.ainvoke([system, user])
        content = getattr(response, "content", None) or str(response)
        return self._extract_action(content)

    def _render_system(self, tools: list[dict]) -> str:
        return self._instructions + "\n" + json.dumps(tools, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_action(content) -> dict:
        """Parse a JSON action object out of arbitrary model output; raise PolicyError on failure."""
        if content is None or not str(content).strip():
            raise PolicyError("policy returned empty output")
        text = _strip_fences(str(content))
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise PolicyError("no JSON object found in policy output", details={"content": text[:200]})
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            raise PolicyError("invalid JSON from policy", details={"error": str(exc), "content": text[:200]})
        if not isinstance(data, dict):
            raise PolicyError("policy output is not a JSON object", details={"content": text[:200]})
        return data


def _strip_fences(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
