"""Real-API integration smoke test: LearnBench + APIPolicy + RolloutRunner.

Mirrors the app's startup (``main.py`` calls ``load_dotenv()``), then drives a few
short rollouts through the production ``ChatOpenAI`` (``graph.get_llm()``). No
secrets are printed or committed — the API key is read from ``backend/.env`` only.

Run from ``backend/``:
    python smoke_test_api.py
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()  # must run BEFORE importing graph (graph builds the app + reads env)

from graph import get_llm  # noqa: E402  (production LLM path; heavy import side effect)
from learnbench import LearnEnv, Task  # noqa: E402
from policy import APIPolicy  # noqa: E402
from rollout import RolloutConfig, RolloutRunner  # noqa: E402


TASKS = [
    Task(id="smoke_easy", target_topic="arithmetic", target_mastery=0.1, total_days=3, minutes_per_day=200),
    Task(id="smoke_medium", target_topic="probability", target_mastery=0.5, total_days=5, minutes_per_day=180),
    Task(id="smoke_hard", target_topic="transformer", target_mastery=0.8, total_days=14, minutes_per_day=180),
]


async def run_one(task: Task, policy: APIPolicy, seed: int) -> dict:
    env = LearnEnv(task=task)
    runner = RolloutRunner(env, policy, RolloutConfig(max_steps=15, max_invalid_actions=5))
    result = await runner.run(seed=seed)
    tr = result.trajectory
    error = None
    if result.termination_reason == "policy_error" and tr.steps:
        err = tr.steps[-1].error or {}
        error = err.get("message", "")[:300]
    return {
        "task": task.id,
        "target": f"{task.target_topic}:{task.target_mastery}",
        "success": result.success,
        "termination": result.termination_reason,
        "invalid_actions": result.invalid_actions,
        "steps": result.total_steps,
        "total_reward": round(tr.final_reward, 4),
        "error": error,
    }


async def main() -> None:
    policy = APIPolicy()  # default_llm() -> graph.get_llm()
    print("=" * 72)
    print("Real-API smoke test: LearnBench + APIPolicy + RolloutRunner")
    print("=" * 72)
    rows = []
    for i, task in enumerate(TASKS, 1):
        print(f"\n[{i}/{len(TASKS)}] running task {task.id} ({task.target_topic} -> {task.target_mastery}) ...")
        row = await run_one(task, policy, seed=42)
        rows.append(row)
        print(f"  success={row['success']}  reason={row['termination']}  "
              f"invalid={row['invalid_actions']}  steps={row['steps']}  reward={row['total_reward']}")
        if row["error"]:
            print(f"  api/policy error: {row['error']}")

    print("\n" + "-" * 72)
    print(f"{'task':14} {'success':8} {'termination':22} {'invalid':8} {'steps':6} {'reward':8}")
    print("-" * 72)
    for r in rows:
        print(f"{r['task']:14} {str(r['success']):8} {r['termination']:22} "
              f"{r['invalid_actions']:8} {r['steps']:6} {r['total_reward']:8}")
    print("-" * 72)


if __name__ == "__main__":
    asyncio.run(main())
