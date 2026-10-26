# Reactive Samplers 小规模实验进度

代码先在 RunPod 修改并验证，再同步至 Trento。当前已提交作业，尚无本次实验的质量或加速结论。

## 配置与数据

- RunPod 项目：`/workspace/home/shencanyu/deepresearch/workspace/reactive-samplers`
- Trento 项目：`/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers`
- 数据：`sampler_runs/datasets/coco_hpd_20_v1/`
- COCO 2017 train captions 和 HPD v2 training prompts 各抽 10 条，选择随机种子 20261026。每个来源各 5 条探索、各 5 条独立冻结评测；提示词和生成种子保存在 JSONL 中。
- SD3-Medium，1024×1024，bfloat16，CFG 7，max_sequence_length 256，官方 FlowMatch Euler，参考时间表 28 步，最多 28 NFE。
- Agent：按最新要求改为 Codex CLI `gpt-6-luna`，`model_reasoning_effort=high`，`danger-full-access`，每步检查、同一会话持续续接。RunPod tmux 会话：`reactive-mini20-agent`。
- 探索每次调用超时：桥接 540 秒，采样端 600 秒。候选导入、同一步分析或 action 校验失败时，现有机制恢复旧策略继续。

## 已实现并验证

- 在线 latent、velocity、预测 clean latent 的统计及历史变化、余弦相似度；4×4 局部空间变化；CFG 条件/无条件差异；中间层特征统计。
- 初始捕获层为 `transformer_blocks.11`。Agent 可以修改 `FEATURE_LAYERS`、组合和删除观察；新增层从下一次模型预测开始捕获。当前层原始输出可在暂停状态中检查，历史保留摘要，避免累计所有层张量。
- 中间 clean latent 的 VAE 解码、CLIP/VLM 辅助诊断按用户确认暂缓。
- 每步 events 记录当前及下一 timestep/sigma、NFE、策略版本和 observation；每图 results 记录 steps、完整 timestep/sigma 序列、Agent 调用次数和耗时。
- HPSv2.1 使用本地完整权重和参考模型/预处理，CLIPScore 使用本地 OpenAI ViT-B/32，分数为 max(100×cosine, 0)。离线实际评分检查通过。
- 已验证 Codex 无沙箱新会话、续接、执行 shell 和写入测试文件。
- 验证环境为已迁移的 torch 2.5.1+cu121 / diffusers 0.35.1；15 项测试通过，包括策略热切换、候选回退、观察值和官方 Euler 一致性。
- 原 `check_environment.sh` 因把 Python 3.12 系统 torch 路径注入另一解释器而失败；随后在正确的 Python 3.11 推理环境完成上述 15 项检查。

## 已提交的 Trento 作业

| Job ID | 内容 | 依赖 |
|---|---|---|
| 48063112 | 10 条探索提示词的默认时间表对照 | 无 |
| 48063185 | 10 条冻结评测提示词的默认时间表对照 | 无 |
| 48063187 | 10 条探索提示词，每步调用 Agent | 48063112 成功 |
| 48063191 | 冻结策略生成 10 条独立评测图，不调用 Agent | 48063187 和 48063185 成功 |

最后核实时两个对照为 Priority 排队，后两项等待依赖；Slurm 未给出预计启动时间。

## 输出与比较

Trento 输出位于 `sampler_runs/mini20-20261026/`。每个 run 下有 config、images、results、feedback、trajectories；探索另有 decisions、workspace、programs 和 frozen_policy。Agent 能读取探索集同 prompt/seed 的默认参考反馈，契约禁止读取独立评测提示词及结果。

默认对照不启用 observation/中间层钩子。最终比较使用独立评测的同 prompt、同 seed 配对，报告 HPSv2.1、CLIPScore、平均 NFE/steps、采样和端到端耗时。小样本不计算 FID，不预设质量容忍阈值。

`sampling_seconds` 包含观察和探索时的等待；`agent_seconds` 单独记录桥接等待/诊断时间。`end_to_end_seconds` 包含每图文本编码与解码，排除质量评分和一次性模型加载。冻结评测无 Agent。

冻结作业成功后自动写 `comparison.json`、`comparison.md`。报告脚本检查 prompt/seed、生成配置配对，并检查冻结评测没有 Agent 调用。尚需核实作业结束状态、真实图片和比较结果；未自动推送 GitHub。
