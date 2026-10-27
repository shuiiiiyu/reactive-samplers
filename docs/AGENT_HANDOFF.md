# Reactive Samplers 接管说明

正式项目：`/workspace/home/shencanyu/deepresearch/workspace/reactive-samplers`（RunPod）。
本文件解释项目和代码，不汇总实验成绩。详细机制见 [three-methods.md](three-methods.md)。

## 研究目标

不训练生成模型，让编码 Agent 在探索过程中修改 Python 采样规则，减少完整去噪计算，
同时保持图片自然、逼真和提示词一致性。探索结束后导出规则，在无 Agent 的情况下评估。

当前只有三种方法，保留原编号：

| 编号 | method | skip=True 的行为 |
|---|---|---|
| 2 | grid_skip | 当前节点不预测、不更新 latent，进入下一个原定 timestep |
| 3 | prediction_cache | 复用上次真实预测的最终 CFG 输出，仍更新当前 latent |
| 4 | residual_cache | 当前输入 feature 加旧主干残差，再运行当前输出头，仍更新 latent |

所有方法都在当前完整预测之前决策。method2 不扫描后续候选节点，不扩大积分区间。
method1 自定义步长已移除。三个方法都只返回 `SkipAction(skip=bool)`。

## 代码结构

- `wrapper.py`：共用隔离分析和三种方法分派。
- `grid_sampling.py`：method2 当前节点的预测/更新或省略。
- `cache_sampling.py`：method3/4 的独立入口及缓存采样循环。
- `models/sd3.py`、`models/sdxl/`：官方 Diffusers 0.35.1 模型与 Euler 适配。
- `models/cache.py`：浅层输入 feature、最终预测缓存、整个主干残差缓存。
- `observations.py`：基础 observation 计算和英文数学定义。
- `context.py`：当前状态、参考状态、SkipAction。
- `programs.py`：策略复制、加载、版本和冻结；源码哈希也作为版本标识。
- `agent.py`、`backends/`：Agent 请求、inspect、反馈和候选策略交接。
- `decision_records.py`：四项英文报告、原始回答和真实执行结果归档。
- `evaluation.py`：CLIPScore、ImageReward、相同 prompt/seed 的参考差值。
- `recovery.py`：无效候选拒绝、运行失败回退、状态隔离、超时和失败记录。
- `gallery.py`、`web/index.html`：RunPod 图片、指标、Agent 记录查看与下载。

每种方法的起始代码在 `prompts/<method>/`，SDXL grid_skip 使用 `sdxl_grid_skip/`。
Agent 修改的是实验自己的 workspace，不是框架起始模板。

## Observation 与参考对象

基础字段为 `latent_rel_change`、`feature_rel_change`、`sigma`、`step_delta_sigma`、
`since_real_delta_sigma`、`consecutive_skips`；method3/4 另有 `has_reference`。

变化量为 `mean(abs(current-reference)) / max(mean(abs(reference)), 1e-8)`。
method2 比较上次真实预测节点的输入；method3/4 比较紧邻上一原定节点的输入，
包括上一节点复用了缓存的情况。首步缺参考时变化量为 null。观测参考和预测缓存分别维护。

浅层 feature 是 SD3 的 pos_embed 或 SDXL 官方缩放/居中后的 conv_in 输出，
不含 timestep 调制。决策时没有当前完整预测或深层输出，不能为判断是否省计算先运行它。
每轮 JSON 与 inspect 都带英文含义、数学定义、参考节点、单位和缺失值解释。
新增信号需要同步填写 `analysis.OBSERVATION_SPEC`。

## 规则、反馈和记录

Agent 必须写“满足什么状态条件就跳过/复用”的完整规则，而不是当前步的一次性开关。
规则必须能从新图片的空 memory 开始运行，不能依赖 Agent 再改一次代码才能继续。
每张图片完成后计算 CLIPScore 与 ImageReward，写入 `results.json` 和 `feedback.json`；
下一次请求的 `new_completed_images` 提供新增反馈。有同 prompt/seed 参考时提供配对差值。
分数是延迟反馈，不能证明某次跳过造成了成图变化，也不是严格逼真度保证。

每轮结束用四项英文标题说明使用的信号、规则修改、理由和新增信号。
`agent_decisions/` 保存 JSON 和 Markdown、原始回答、接受/拒绝/回退及实际动作。
网站直接读取这些记录；缺失或不完整报告会标明，不会生成虚假的 Agent 说明。

## 配置与运行

`conf/sd3_main.yaml` 是公共默认配置，SD3 原始步数在 `reference_steps`，默认 28。
`conf/sdxl_main.yaml` 覆盖 SDXL 模型、精度、CFG 和步数（默认 30）。
默认指标是 `[clip_score, image_reward]`。HPS 不再进入正式评分与反馈。
模型路径、预算、reuse 上限、Agent 调用计划也在配置中。
已生成的实验 JSON 是独立快照，修改 YAML 不会自动改写它们。

模式为 default / explore / evaluate。evaluate 使用带 manifest 的冻结策略，不调用 LLM。
每图重置策略 memory、观测参考和缓存；探索图之间保留策略源码及 Agent 会话。
旧 method2 策略对应旧执行语义，需重新探索、验证，不能当作当前方法的结果。

## 资源与边界

基础权重在 models/，上游 baseline 在 baselines/，数据及相关资源在 benchmarks/，
指标在 metrics/，生成成果在 sampler_runs/。不要编辑导入的上游 source/。
CLIP 与 ImageReward 使用本地权重，ImageReward 的兼容依赖见 pyproject.toml 与 uv.lock。
保留已安装环境、模型、数据、图片、最终策略和 Agent 正式记录。
不重新添加配环境或烟雾测试脚本。CPU 回归用现有环境，在 tmux 中执行：
`python -m unittest discover -s tests -v`。
GPU 生成或付费 Agent 调用需要用户授权；CPU 测试不构成质量或加速结论。
