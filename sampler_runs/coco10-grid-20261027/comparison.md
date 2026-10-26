# COCO 十样本跳步对比实验

同一批 COCO Train prompt 与 seed；Luna / low / 标准速度。属于在线探索对照，不能作为独立冻结泛化评测。
总耗时包含 Agent 等待；模型前向耗时单独记录。回退样本保留并明确标记，不能混入干净方法对比。

| 组别 | 已完成 | 平均 HPSv2.1 | 平均 CLIPScore | 平均 NFE | 平均前向秒数 | 平均总秒数 | 平均 Agent 秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sd3_default | 0/10 | — | — | — | — | — | — |
| sd3_skip | 0/10 | — | — | — | — | — | — |
| sdxl_default | 0/10 | — | — | — | — | — | — |
| sdxl_skip | 0/10 | — | — | — | — | — | — |

## SD3 配对结果

完整配对 0/10；无失败/回退的配对 0/10。

## SDXL 配对结果

完整配对 0/10；无失败/回退的配对 0/10。

逐图配置、timestep 和保留/跳过索引见 comparison.csv；逐步 action、策略版本、Agent 最终输出见 decision_trace.json。
Agent 输出是该次代码修改/保留的原文说明；实际逐候选 action 以 grid_decisions 为准。
