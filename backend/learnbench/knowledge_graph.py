"""Small synthetic prerequisite graph for LearnBench.

A ``Topic`` records an id, a difficulty in ``[0, 1]``, an estimated study time
and its direct prerequisites. ``KnowledgeGraph`` validates that the graph is
well-formed (known prerequisites, no cycles) and answers prerequisite queries
without an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import MASTERED_THRESHOLD


@dataclass(frozen=True)
class Topic:
    id: str
    difficulty: float  # 0 = trivial, 1 = hardest
    study_minutes: int  # rough minutes to reach full mastery
    prerequisites: tuple[str, ...] = ()


class KnowledgeGraph:
    def __init__(self, topics: dict[str, Topic]):
        self._topics: dict[str, Topic] = dict(topics)
        self._validate()

    # -- validation ---------------------------------------------------------
    def _validate(self) -> None:
        ids = set(self._topics)
        for topic in self._topics.values():
            if not 0.0 <= topic.difficulty <= 1.0:
                raise ValueError(f"topic {topic.id!r} difficulty out of [0,1]: {topic.difficulty}")
            if topic.study_minutes <= 0:
                raise ValueError(f"topic {topic.id!r} study_minutes must be > 0")
            for prereq in topic.prerequisites:
                if prereq not in ids:
                    raise ValueError(f"topic {topic.id!r} has unknown prerequisite {prereq!r}")
        self._ensure_acyclic()

    def _ensure_acyclic(self) -> None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {topic_id: WHITE for topic_id in self._topics}

        def visit(topic_id: str, path: list[str]) -> None:
            color[topic_id] = GRAY
            for prereq in self._topics[topic_id].prerequisites:
                if color[prereq] == GRAY:
                    raise ValueError("prerequisite cycle: " + " -> ".join(path + [prereq]))
                if color[prereq] == WHITE:
                    visit(prereq, path + [prereq])
            color[topic_id] = BLACK

        for topic_id in self._topics:
            if color[topic_id] == WHITE:
                visit(topic_id, [topic_id])

    # -- queries ------------------------------------------------------------
    def topic_ids(self) -> list[str]:
        return list(self._topics.keys())

    def has_topic(self, topic_id: str) -> bool:
        return topic_id in self._topics

    def get(self, topic_id: str) -> Topic:
        return self._topics[topic_id]

    def direct_prerequisites(self, topic_id: str) -> list[str]:
        return list(self._topics[topic_id].prerequisites)

    def unmet_prerequisites(
        self, topic_id: str, mastery: dict[str, float], threshold: float = MASTERED_THRESHOLD
    ) -> list[str]:
        return [
            prereq
            for prereq in self._topics[topic_id].prerequisites
            if mastery.get(prereq, 0.0) < threshold
        ]

    def prerequisites_satisfied(
        self, topic_id: str, mastery: dict[str, float], threshold: float = MASTERED_THRESHOLD
    ) -> bool:
        return not self.unmet_prerequisites(topic_id, mastery, threshold)


_DEFAULT_TOPICS: dict[str, Topic] = {
    # --- foundations ---
    "arithmetic": Topic("arithmetic", 0.10, 20),
    "algebra": Topic("algebra", 0.30, 40, ("arithmetic",)),
    "functions": Topic("functions", 0.40, 40, ("algebra",)),
    "calculus": Topic("calculus", 0.60, 60, ("functions",)),
    # --- linear algebra track ---
    "matrix_multiplication": Topic("matrix_multiplication", 0.40, 40, ("algebra",)),
    "linear_transformation": Topic("linear_transformation", 0.50, 50, ("matrix_multiplication",)),
    "eigenvectors": Topic("eigenvectors", 0.60, 50, ("linear_transformation",)),
    "vector_similarity": Topic("vector_similarity", 0.50, 40, ("linear_transformation",)),
    # --- probability / statistics track ---
    "probability": Topic("probability", 0.40, 40, ("algebra",)),
    "random_variables": Topic("random_variables", 0.50, 50, ("probability",)),
    "statistics": Topic("statistics", 0.50, 50, ("probability",)),
    "hypothesis_testing": Topic("hypothesis_testing", 0.60, 60, ("statistics",)),
    # --- optimization / ML track ---
    "gradient_descent": Topic("gradient_descent", 0.60, 50, ("calculus", "linear_transformation")),
    "regression": Topic("regression", 0.50, 50, ("linear_transformation", "statistics")),
    "classification": Topic("classification", 0.55, 50, ("regression", "probability")),
    "neural_networks": Topic("neural_networks", 0.70, 70, ("regression", "gradient_descent")),
    "deep_learning": Topic("deep_learning", 0.75, 80, ("neural_networks",)),
    # --- attention / transformers ---
    "attention": Topic("attention", 0.70, 60, ("vector_similarity",)),
    "transformer": Topic("transformer", 0.85, 90, ("attention", "deep_learning")),
    # --- misc ---
    "optimization": Topic("optimization", 0.65, 60, ("gradient_descent",)),
    "nlp_basics": Topic("nlp_basics", 0.55, 50, ("probability", "vector_similarity")),
    "reinforcement_learning": Topic(
        "reinforcement_learning", 0.80, 90, ("neural_networks", "probability")
    ),
}


def build_default_graph() -> KnowledgeGraph:
    """Return the built-in 22-topic prerequisite graph (a DAG, in topological order)."""
    return KnowledgeGraph(_DEFAULT_TOPICS)
