# RunPod 失败回退实现

状态：代码已在 RunPod 实现并通过 35 项 CPU 故障注入和回归测试；未同步 Trento，未重启正在服务现有实验的桥接进程。测试未调用真实 LLM，也未提交 GPU 作业。

项目位置：`/workspace/home/shencanyu/deepresearch/workspace/reactive-samplers`

## 实现内容

- 将采样步骤分为尝试和提交。analysis/updater 使用隔离的当前及历史状态，memory 只在 action 与新 latent 均通过检查后提交。
- 复用当前预测恢复本步，不重新计算模型。恢复 Scheduler 的 sigmas、timesteps、step/begin index 和失败尝试消耗的 RNG；已完成的前序步骤保留。
- 策略连续成功提交两步后才具备回退资格。按版本整体恢复 analysis、updater、辅助模块；不同代码版本的 memory 重置。未通过试运行的版本不会升级为回退目标。
- 可用版本失败时继续尝试更早版本，最后使用不可由 Agent 修改的框架默认规则沿官方参考时间表继续。版本内容哈希不匹配时跳过并报告。
- 覆盖导入/语法/SystemExit、诊断错误、后续运行期错误、非法 observation/action/层配置、NaN latent 等。
- Linux 主线程中的策略导入、analysis 和 updater 有 Python 执行超时；默认 30 秒，可通过配置 `policy_timeout_seconds` 设置采样规则调用上限。
- CLI 调用超时终止整个工具进程组。不完整或失败的 JSONL turn 不再当作成功修改。
- 新邮箱协议将修改暂存到 decision/candidate，通过请求编号、基础版本、候选哈希和当前步骤验证后再发布。迟到答复不能覆盖运行中的工作区；旧协议保留兼容能力。
- 失败报告、上一轮修改结局和最近错误提供给下一轮 Agent；完整失败报告与日志也会转到桥接端可读的位置。
- 冻结评测发生回退时标记该图片不适合干净方法比较；保留所有图片与单独 clean_summary，配对比较禁止无提示混入默认接管样本。

## 代码位置

均相对于上面的 RunPod 项目根目录：

- `reactive_samplers/recovery.py`：状态隔离、Python 超时、试运行/回退版本、失败记录、崩溃现场。
- `reactive_samplers/wrapper.py`：每步事务、候选验收、运行期恢复及实际结果标记。
- `reactive_samplers/programs.py`：不可变版本校验、恢复和内置默认规则。
- `reactive_samplers/models/sd3.py`：Scheduler 状态快照/恢复、预测次数记录；官方求解公式未改。
- `reactive_samplers/agent.py`：诊断隔离、上一轮结局、失败反馈。
- `reactive_samplers/backends/__init__.py` 与 `scripts/runpod_codex_bridge.py`：调用超时、进程组清理、候选暂存和迟到答复处理。
- `docs/failure-recovery.md`：详细行为与边界。

## 运行产物

- `failures.jsonl`、`failure_report.md`。
- `decisions/<id>/outcome.json` 与保留的候选/被拒代码。
- 中断时的 `trajectories/<sample>/failure.json`，以及可保存时的 `failure-state.pt`。
- 每图结果中的 `failure_count`、`recovered`、`recovery_steps`、`default_fallback_steps`、`retry_nfe`、`valid_for_benchmark`。

## 验证

RunPod 测试日志：`sampler_runs/setup/logs/recovery-verified.log`。

结果：35 项通过，包含现有热切换测试、同一步邮箱协议集成、后续步骤报错、未通过试运行版本、连续回退失败、版本损坏、状态污染、真实 torch 张量隔离、Python 死循环超时、CLI 子进程清理及官方 Diffusers Euler 一致性。

## 边界

策略仍是可信 Python。SIGKILL、原生崩溃、无法返回的底层调用或 CUDA 上下文损坏，不能保证原进程继续；能够捕获时保存现场并明确中断。信号超时仅在 Linux 主线程启用。自动从现场恢复 Slurm 作业尚未实现。

本次实现不会撤销此前已经成功提交的图片轨迹，也不会根据 HPS/CLIPScore 下降自动回退。35 项检查不等于已完成新版框架的真实 GPU/远程 Agent 验收；当前 Trento 实验仍使用旧部署。
