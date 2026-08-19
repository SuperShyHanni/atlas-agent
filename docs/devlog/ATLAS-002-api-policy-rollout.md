# ATLAS-002 — Implement API Policy Integration + Rollout + Trajectory Recording

## Task ID

`ATLAS-002`

## Title

Implement API Policy Integration + Rollout + Trajectory Recording（Phase 2）

## Goal

把现有 API 模型接入 LearnBench（不训练）：实现策略抽象 `BasePolicy` + `APIPolicy`、rollout 执行器与逐步轨迹记录；严格解析模型输出、失败不崩溃、防死循环；仍不改动现有 ATLAS 产品行为。

## Scope

**做**

- `BasePolicy` 抽象基类（`async act(observation, tools) -> dict`）
- `APIPolicy`（复用 `graph.get_llm()` 的 OpenAI 兼容路径，非 LocalQwen）
- 结构化动作解析闭环（剥离 markdown fence、定位首个 JSON、`json.loads`，失败抛 `PolicyError`）
- rollout runner（observation → action → env.step 循环，直至终止或超步）
- 逐步轨迹记录（task_id / seed / observation / action / result / reward 分解 / error / final_reward / success / termination_reason）
- 防死循环与失败保护（`max_steps` / `max_invalid_actions`，无效动作不崩溃）
- 单元测试

**不做**

- `LocalQwenPolicy` / 本地 Qwen 推理
- SFT / GRPO / TRL / vLLM / CUDA 重型依赖
- 轨迹落盘 / 数据集导出（Phase 3）
- 对现有 `graph.py` / agents / tools / frontend 的任何改动

## Design

- 新增 `backend/policy/` 与 `backend/rollout/` 兄弟包，rollout 只依赖 `BasePolicy` 抽象，与具体策略解耦
- `APIPolicy` **惰性 import** `graph.get_llm` 与 `langchain_core`，保证 `policy` / `rollout` 模块离线可导入（测试不拉 LangGraph / chromadb）
- 失败分两类：策略抛异常 → `policy_error`（立即终止）；动作解析失败 → `invalid_actions` 计数（可容忍至上限）
- 环境仍是唯一事实来源，rollout 只记录、不直接改状态

## Changes

全部新增 7 个文件，修改 1 个文件。

**`backend/policy/`**

```
__init__.py      # 导出 BasePolicy / PolicyError / tool_schema / APIPolicy / default_llm
base.py          # BasePolicy(ABC) + PolicyError + tool_schema()（由 TOOL_SPECS 生成）
api_policy.py    # APIPolicy + default_llm()（惰性 graph.get_llm）+ _extract_action 静态解析
```

**`backend/rollout/`**

```
__init__.py      # 导出 RolloutConfig / RolloutResult / RolloutRunner
runner.py        # RolloutRunner.run + RolloutConfig + RolloutResult
```

**`backend/tests/`**

```
test_policy.py   # tool_schema / _extract_action 离线测试 + 可选 live 测试（gated by OPENAI_API_KEY）
test_rollout.py  # rollout / trajectory 离线测试（ScriptedPolicy / ErrorPolicy 假件）
```

**修改**

```
backend/learnbench/trajectory.py   # TrajectoryStep 新增 error: dict | None 字段（纯增量、向后兼容）
```

## Validation

`python -m unittest discover -s tests -t . -v`

- **Ran 47 tests, OK (skipped=1)**
- Phase 1 遗留 30 个全部通过
- Phase 2 新增 16 个通过 + 1 个跳过（live API 测试因 `OPENAI_API_KEY` 未设置被正确 skip）
- rollout 覆盖：`reaches_target` / `out_of_time` / `max_steps` / `too_many_invalid_actions` / `invalid_then_valid` / `policy_error` / `trajectory_json_roundtrip` / `final_reward_is_sum`

## Issues

- `APIPolicy.act` 端到端路径依赖真实 LLM，未在默认套件验证（由 gated 测试覆盖）；解析健壮性已离线充分测试
- rollout 层 `max_rollout_steps`（默认 50）与 env 层 `max_steps`（默认 200）是两层独立上限，语义略重叠但职责不同
- 轨迹仅内存返回，尚未落盘 / 导出数据集

无阻塞问题。

## Decision

1. `APIPolicy` 惰性 import，保持离线可导入、测试零外部依赖
2. 策略与执行解耦，测试用 `ScriptedPolicy` / `ErrorPolicy` 假件，不依赖 API
3. 策略异常与解析失败分开计数：前者立即终止，后者可容忍至上限
4. 轨迹 schema 保持 `1.0`，仅增量新增 `error` 字段，不破坏既有序列化
5. 默认 `max_steps=50` / `max_invalid_actions=10`，双重防死循环

## Status

Done（改动未提交）

## Next

Phase 3 — 轨迹数据集构建与本地基线前数据处理：

- 多次 rollout 结果批量落盘为 JSONL 数据集
- `Trajectory` 批量序列化 + 统计聚合
- 为后续 SFT / GRPO 语料准备

仍不引入本地模型 / 训练依赖。

## Related

- [ATLAS-001-learnbench-mvp.md](ATLAS-001-learnbench-mvp.md)
- 对应 `AGENTS.md` §11「Policy abstraction」/ §12「Phase 2 — Trajectory recorder and API teacher」
