# ATLAS-001 — Implement LearnBench MVP

## Task ID

`ATLAS-001`

## Title

Implement LearnBench MVP（Phase 1）

## Goal

为 ATLAS 搭建训练导向研究子系统的第一块地基：实现一个**完全离线、无 LLM** 的合成学习规划环境 LearnBench，作为后续 SFT / GRPO 的训练与评测底座。当前 ATLAS 产品路径保持原样不动。

## Scope

**做**

- 独立 `StudentLearningState`
- 结构化 `AgentAction` + 观察 schema
- 前置知识图谱 + 硬约束守卫
- 规则化模拟器
- `LearnEnv.reset` / `LearnEnv.step`
- 可验证的奖励分解
- 单元测试

**不做**

- 本地 Qwen / SFT / GRPO / TRL / vLLM / CUDA 重型依赖
- 轨迹自动录制器（Phase 2）
- `APIPolicy`
- 对现有 `graph.py` / agents / tools / main 等产品代码的任何改动

## Design

新增 `backend/learnbench/` 包，采用包内相对导入（与现有 `agents/`、`tools/` 一致）。

- 纯函数奖励分量 + 规则化模拟器 + 严格动作校验 + 结构化错误（不崩溃）
- `LearnEnv` 提供 Gym 式 `reset` / `step` 接口
- 环境是唯一事实来源，策略不直接改状态

## Changes

全部新增，未改动任何现有文件（`git status` 仅有 untracked 新增）。

**`backend/learnbench/`**

```
__init__.py          # 包公共 API 导出
errors.py            # LearnBenchError + 稳定错误码（结构化错误）
state.py             # StudentLearningState + MASTERED_THRESHOLD
action.py            # AgentAction + 5 工具 ArgumentSpec + parse_action 校验
observation.py       # build_observation（只暴露允许的信息）
knowledge_graph.py   # Topic + KnowledgeGraph + 22 主题默认 DAG（含环检测）
constraints.py       # ensure_* 硬约束守卫（时间/话题/天数/mastery/task）
simulator.py         # 规则化学习转移 + 测验评分（无 LLM）
reward.py            # RewardWeights + RewardBreakdown + 纯函数奖励分量
tasks.py             # Task + default_task / sample_tasks
trajectory.py        # 带版本号的 Trajectory 序列化
env.py               # LearnEnv.reset / step
```

**`backend/tests/`**

```
__init__.py
test_learnbench.py   # 30 个单元测试
```

## Validation

`python -m unittest discover -s tests -t . -v`

- **30 个测试全部通过（0.010s）**
- 覆盖 AGENTS.md §13 最小集：`state_bounds` / `time_budget` / `invalid_topic` / `prerequisite_violation` / `quiz_reproducibility` / `reset_seed` / `env_step` / `action_validation` / `termination` / `trajectory_serialization`
- 1000 步随机合法/非法动作模糊测试：无崩溃、mastery ∈ [0,1]、时间非负
- 固定种子下轨迹逐值复现
- AGENTS.md §6 手动示例可运行

## Issues

- 环境未安装 `pytest`，改用标准库 `unittest`（零新依赖，`python -m unittest` 即可跑）
- backend 根不是 Python 包，测试文件用 `sys.path` bootstrap 保证可独立执行

无阻塞问题。

## Decision

1. 用 `unittest` 而非引入 pytest
2. 前置违规采用**软惩罚**：允许学习但效率减半 + `prerequisite < 0`（符合「缺前置降低效率」而非硬拒绝）
3. `take_quiz` 零分钟、不改 mastery（纯测量）
4. 奖励权重用默认值、不调参
5. `Trajectory` 先做独立带版本 schema（`schema_version="1.0"`），录制器接入留 Phase 2

## Status

Done

## Next

Phase 2 — Trajectory recorder + API teacher：

- 实现 `BasePolicy` + `APIPolicy`（复用 `graph.py` 的 `ChatOpenAI`）
- 结构化动作解析闭环
- 轨迹录制器与 rollout runner（步数上限、终止原因落盘）

仍不引入本地模型 / 训练依赖。

## Related

- 无 PR / commit（改动未提交）
- 对应 `AGENTS.md` §6 / §12「Phase 1 — LearnBench MVP」及 §17 完成定义
