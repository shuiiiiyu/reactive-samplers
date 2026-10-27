# Reactive Samplers 项目说明与代码交接

本文只解释 **RunPod 上的项目**：研究目标、目录结构、四种采样机制、策略接口和配置入口。

## 1. 项目位置与阅读顺序

RunPod 项目根目录：

```text
/workspace/home/shencanyu/deepresearch/workspace/reactive-samplers
```

GitHub 仓库：<https://github.com/vibe-hust/reactive-samplers>

本地聊天工作目录 `C:\Users\24778\Documents\ChatGPT\agent+diffusion` 包含修改副本和辅助文件；接管时以 RunPod 项目源码为准。

建议依次阅读：

1. `AGENTS.md`：项目约定和允许修改的范围。
2. `conf/sd3_main.yaml`：默认配置。
3. `reactive_samplers/context.py`：状态与动作的数据结构。
4. `reactive_samplers/wrapper.py`：统一入口及方法①②。
5. `reactive_samplers/cache_sampling.py`：方法③④的预测前决策循环。
6. `reactive_samplers/models/cache.py`：两种缓存的实际内容、模型边界与复用计算。
7. 各方法的 `prompts/analysis.py`、`updater.py`、`sampling_agent.md`。
8. `reactive_samplers/agent.py`、`programs.py`：Agent 如何改代码、加载和保留策略。

下文相对路径均以 RunPod 项目根目录为基准。

## 2. 研究目标：让 Agent 发现采样计算规则

项目研究冻结扩散模型权重时，能否依据采样过程中的当前信息与历史信息，自适应分配计算，减少模型预测或改变采样步长。

项目中有两层决策，必须区分：

- **编码 Agent**：读取 observation 和反馈，修改 Python 分析程序及更新程序。
- **Python 策略**：采样运行时执行 `analyze()` 和 `choose()`，输出下一 sigma 或 keep/skip。

不是每个 timestep 都由 LLM 直接返回一个布尔动作。Agent 是否介入由 schedule 控制；冻结评估只运行 Python 策略，不调用 LLM。

项目不通过训练模型、改变 CFG、改变精度或修改数值求解公式来获得方法收益。四种方法的区别在于采样控制或缓存复用机制。

模型支持范围：

| 模型 | 模型结构 | Scheduler | 可用方法 |
|---|---|---|---|
| SD3 Medium | Transformer | 官方 `FlowMatchEulerDiscreteScheduler` | ①②③④ |
| SDXL Base | U-Net | 官方 `EulerDiscreteScheduler`，当前适配器要求 epsilon prediction | ②③④ |

不能把 SDXL 当前预测统一称为 velocity；它在当前适配器中是 epsilon。

## 3. 目录结构

```text
reactive_samplers/
  context.py                 Context、Prediction、Action、SkipAction
  wrapper.py                 统一 sample() 入口；方法①②及预测后决策循环
  cache_sampling.py          方法③④及预测前决策循环
  models/
    sd3.py                   SD3 文本编码、模型预测、CFG、Scheduler 与解码适配
    sdxl/__init__.py         SDXL 对应适配与输入缩放
    cache.py                 浅层 feature、缓存刷新、最终预测复用、主干残差复用
  agent.py                   Agent 暂停点、快照、请求、重复 inspect、策略更新记录
  programs.py                策略源码加载、版本保存、恢复、冻结导出
  recovery.py                策略失败回退、状态隔离、超时、错误记录
  schedule.py                every_step / checkpoints 的 Agent 介入计划
  backends/__init__.py       命令行 Agent 与 mailbox 后端
  prompts/
    analysis.py              ①的初始 observation 程序
    updater.py               ①的初始 sigma 选择程序
    sampling_agent.md         ①的 Agent 契约
    grid_skip/               ②的 SD3 模板及契约
    sdxl_grid_skip/           ②的 SDXL 模板及契约
    prediction_cache/        ③的模板及契约
    residual_cache/          ④的模板及契约
  cli.py                     配置执行、inspect / freeze / report 命令
  evaluation.py              质量指标适配
  report.py                  事件与统计写入
  gallery.py                 结果浏览与生成请求服务
  generation.py              网页独立生成流程，不是四方法主入口
  web/index.html             网页界面

conf/
  sd3_main.yaml              SD3 默认参数
  sd3_grid_skip.yaml         继承 SD3 默认参数，仅切换为 grid_skip
  sdxl_main.yaml             继承公共参数，覆盖 SDXL 模型参数

scripts/
  run_main.py                Hydra 主入口
  run_config_json.py         读取已生成的 JSON，调用同一个 run_config()
  prepare_coco10_grid.py     从 YAML 默认配置准备新的配对实验 JSON
  report_coco10_grid.py      配对报告整理
  runpod_codex_bridge.py     编码 Agent 的 mailbox 桥接
  agent_inspect_helper.py    Agent 反复检查当前暂停状态的辅助入口
  download_models.*          模型获取
  download_research_assets.* 数据、基线与指标资源获取

tests/                       CPU 回归测试
models/                      模型权重；与 reactive_samplers/models/ 的适配代码不同
baselines/                   第三方基线源码及其资源
benchmarks/                  数据、指标相关代码与资源
metrics/                     共用指标资源
sampler_runs/                配置快照、策略版本、轨迹、图片及运行记录
docs/                        项目说明
```

`scripts/` 中还保留其他环境的辅助启动脚本；它们不改变四种方法的核心机制。本文不展开其部署。

## 4. 四种方法：必须保持语义分离

| 编号 | method | 决策发生在何时 | 动作 | 当前网格节点是否更新 latent |
|---|---|---|---|---|
| ① | `adaptive_sigma` | 当前模型预测之后 | 返回任意合法下一 sigma | 按选定跨度更新 |
| ② | `grid_skip` | 当前模型预测之后，判断后续候选节点 | keep/skip 后续原始节点 | 被移除节点不单独更新 |
| ③ | `prediction_cache` | 当前完整模型预测之前 | 重算或复用旧最终预测 | 每个原始区间都更新 |
| ④ | `residual_cache` | 当前昂贵主干计算之前 | 重算或复用旧主干残差 | 每个原始区间都更新 |

统一入口为 `wrapper.sample(..., method=...)`，分发到四个独立、带注释的函数。

### 4.1 方法①：Agent 自定义步长

入口：`wrapper.sample_adaptive_sigma()`。

```text
当前 latent → 真实模型预测 → analysis → updater 返回下一 sigma
           → 检查 sigma 合法性 → 官方 Scheduler 更新
```

策略接口：

```python
def choose(observation, ctx, memory):
    return Action(next_sigma=...)
```

下一 sigma 是 **Scheduler shifting 后的物理 sigma**，必须严格下降且不越过终点。预算用尽时，最后一次预测必须更新到终点。适配器不重复应用 shift。

默认模板沿参考时间表前进；允许自定义连续 sigma，不等于默认模板已经产生任意步长策略。

### 4.2 方法②：直接移除 timestep

入口：`wrapper.sample_grid_skip()`；核心候选扫描函数为 `_choose()`。

```text
在节点 A 真实预测 → 依次检查后续候选 B、C、D
                → B/C 跳过，D 保留
                → 使用 A 的预测，一次从 A 更新到 D
```

策略接口：

```python
def choose(observation, ctx, memory):
    return SkipAction(skip=True)  # 跳过当前正在检查的后续候选节点
```

这里 `ctx.sigma` 对应已有真实预测的源节点；`candidate_sigma` 对应待判断的后续节点。不是先计算 B 的完整预测再决定是否跳过 B。

同一次真实预测可以用于检查多个后续候选节点；这些候选没有各自的新模型观察。终点是积分边界，不需要额外的去噪预测。

①②共用 `_sample_postprediction()`，原有机制保留。

### 4.3 方法③：最终预测缓存

入口：`cache_sampling.sample_prediction_cache()`。

缓存的是最近一次真实预测经过 CFG 等处理后、传给 `scheduler.step()` 的张量：SD3 为 flow velocity；当前 SDXL 适配器为 epsilon。

```text
到达当前节点 → 提取廉价入口 feature → 判断 keep/skip
  keep：运行完整模型，刷新最终预测缓存
  skip：直接使用最近一次真实预测
→ 使用当前 latent、当前 timestep/sigma，执行当前原始区间的 Scheduler 更新
```

实现函数：`models/cache.py::reuse_prediction()`。

跳过时不调用当前去噪模型或输出头。仍然使用最新 latent，不能把旧 latent 当作当前 latent。当前 Scheduler 自行解释该预测，不缓存并套用旧步的 Scheduler 转换结果。

### 4.4 方法④：整个主干残差缓存

入口：`cache_sampling.sample_residual_cache()`。

把模型分成：

```text
latent → 浅层入口得到 H → 昂贵主干得到 Z → 输出头得到最终预测
```

真实计算时缓存：

\[
R_{\mathrm{cache}}=Z_{\mathrm{real}}-H_{\mathrm{real}}.
\]

复用时近似为：

\[
\hat Z_k=H_k+R_{\mathrm{cache}},\qquad
\hat p_k=\operatorname{Head}_k(\hat Z_k).
\]

**`H + R` 是输出头之前的特征，不是最终 velocity。** 当前输入特征仍会影响近似预测。

模型边界在 `models/cache.py` 中明确实现：

| 模型 | H：缓存段输入 | Z：缓存段输出 | 复用时仍执行的输出头 |
|---|---|---|---|
| SD3 | `pos_embed` 输出 | 全部 Transformer blocks 之后、`norm_out` 之前 | 当前 `time_text_embed`、`norm_out`、`proj_out`、unpatchify |
| SDXL | 官方输入缩放及可选中心化后的 `conv_in` 输出 | 整个 down/mid/up U-Net 之后、`conv_norm_out` 之前 | norm、activation、`conv_out` |

SD3 输出归一化继续接收当前时间条件。SDXL 末端输出头不含时间条件，旧 U-Net 残差保留旧时间和文本影响；这是方法近似的一部分。

`refresh_prediction()` 用 hook 捕获 Z，并在真实模型预测中复用已算出的入口 feature，避免重复计算入口。`reuse_residual()` 只运行入口之后的残差加法和输出头，不执行主干。

③④共用 `_sample_cache()`；缓存如何产生近似预测由独立函数区分，不通过修改 Scheduler 公式实现。

## 5. 两个缓存版本使用的 observation

当前默认只使用以下六项：

| 字段 | 含义 |
|---|---|
| `sigma` | 当前候选节点的物理 sigma |
| `step_delta_sigma` | 当前节点到下一原始网格节点的 sigma 距离 |
| `since_real_delta_sigma` | 距最近一次真实预测节点的 sigma 距离 |
| `consecutive_skips` | 上次真实预测以来连续复用多少次 |
| `has_reference` | 是否已有有效预测和参考 feature |
| `feature_rel_change` | 当前入口 feature 相对最近一次真实计算时入口 feature 的变化 |

feature 来自 SD3 的 `pos_embed` 或 SDXL 的 `conv_in`，是图像 latent 的浅层表示，不是文本 embedding，也不是当前深层输出。

\[
\operatorname{feature\_rel\_change}
=\frac{\operatorname{mean}|F_k-F_r|}
{\max(\operatorname{mean}|F_r|,10^{-8})}.
\]

其中 r 是最近一次 **真实计算节点**，不一定是紧邻的上一个网格节点。先对完整张量计算差异，再压成标量，不先做 token/空间均值池化。

已决定不加入 `feature_path_change` 累计量。当前浅层 feature 未融合 timestep modulation，时间信息通过 sigma 字段提供；这不是完整 TeaCache 判据的复现。

③④决策时 `ctx.prediction.guided` 为 `None`；不能为获得当前 velocity、clean latent 或末层 feature 而先运行完整模型。`FEATURE_LAYERS` 默认并要求为空，当前浅层 feature 已由框架提供。

①②仍使用它们各自已有的 observation 模板，没有被这次六字段设计替换。

## 6. 缓存生命周期、动作与预算

每张图开始时清空：

```python
cached_prediction = None
cached_residual = None
reference_feature = None
reference_sigma = None
consecutive_skips = 0
```

在③④中，同一个动作类型具有以下含义：

```python
SkipAction(skip=False)  # 当前步真实计算并刷新缓存
SkipAction(skip=True)   # 当前步复用缓存，仍更新 latent
```

首步强制真实预测；连续复用上限默认 `max_consecutive_skips: 4`。预算逻辑为后续必需刷新预留真实预测次数，可能强制复用；无法满足刷新上限的预算会直接报错。

缓存由框架持有，和 Agent 的策略 memory 分开。数值更新成功后才提交新缓存；策略回退重置策略 memory，不丢弃已提交的历史缓存。

初始 updater 对所有步骤都返回 `skip=False`。这是提供给 Agent 的起点，不是已经选择好的加速阈值；如果没有修改策略且没有预算强制，默认不会发生复用。

## 7. Agent 如何修改代码

框架模板放在 `reactive_samplers/prompts/`，每次运行复制到：

```text
sampler_runs/<run>/
  workspace/analysis.py       当前可编辑分析程序
  workspace/updater.py        当前可编辑决策程序
  programs/<version>/         保存的完整策略源码版本
  decisions/<id>/             暂停快照、请求、Agent 回答、耗时及更新状态
  trajectories/<sample>/      逐步状态与动作记录
  frozen_policy/              冻结导出的策略
```

两个函数的契约：

```python
def analyze(ctx, memory):
    return {...}  # 可序列化的 observation 字典

def choose(observation, ctx, memory):
    return Action(...)  # ①
    # 或 SkipAction(...)，用于②③④
```

`AgentSession.check()` 暂停采样并允许反复检查同一个状态。修改后的策略重新加载，仍在同一个暂停点重新分析和决策；`inspect` 本身不补算尚不存在的当前模型输出。

Agent 修改的是运行目录的 `workspace/` 副本，不应直接修改框架模板或已完成运行的策略记录。各方法有独立契约与 method 标记，冻结导出后不能混用方法。

`Programs` 保存版本并加载 Python 源码；源码哈希用于版本标识，当前代码仍保留冻结导出和 mailbox 的完整性校验。`recovery.py` 负责失败隔离、回退和记录。不要把接管项目理解为允许随意删掉这些现有机制。

## 8. 配置在哪里改

SD3 默认配置：

```text
/workspace/home/shencanyu/deepresearch/workspace/reactive-samplers/conf/sd3_main.yaml
```

关键参数：

```yaml
mode: default
model: sd3
method: adaptive_sigma
device: cuda
dtype: bfloat16
height: 1024
width: 1024
guidance_scale: 7.0
max_sequence_length: 256
reference_steps: 28
max_nfe: ${reference_steps}
max_consecutive_skips: 4
```

`reference_steps` 是原始时间表长度；`max_nfe` 是真实主干预测预算。默认随步数变化，也可以单独设小。

`sd3_grid_skip.yaml` 继承 SD3 配置，仅设置 `method: grid_skip`。`sdxl_main.yaml` 覆盖模型路径、float16、CFG 5、30 步和 SDXL 方法默认值。

从 RunPod 项目根目录可以只查看配置，不加载模型：

```bash
uv run python scripts/run_main.py --cfg job --resolve reference_steps=40
uv run python scripts/run_main.py --config-name sdxl_main --cfg job --resolve
```

以下展示方法切换语法，**执行不带 `--cfg job` 的命令会实际运行推理**：

```bash
uv run python scripts/run_main.py method=prediction_cache reference_steps=40
uv run python scripts/run_main.py method=residual_cache reference_steps=40
```

三个运行模式：

- `default`：不调用 Agent，运行默认或指定策略。
- `explore`：允许 Agent 修改策略，使用指定 backend 和介入 schedule。
- `evaluate`：运行冻结策略，不调用 Agent。

`scripts/prepare_coco10_grid.py` 从 YAML 读取模型默认参数，并生成新的 JSON 配置快照。RunPod 路径示例：

```bash
uv run python scripts/prepare_coco10_grid.py \
  --name coco10-next \
  --runtime-root /workspace/home/shencanyu/deepresearch/workspace/reactive-samplers
```

旧 JSON 是固定快照，不会随 YAML 自动更新。已有尚未运行的 JSON 若要改变步数，应明确修改其中 `reference_steps`、`max_nfe` 并选新输出目录；或直接准备一个新运行。旧 JSON 中的绝对路径需要适合当前 RunPod 环境。

网页独立生成使用 `generation.py` 的请求参数和默认值；修改主实验 YAML 不等于修改所有网页生成请求。

## 9. 读代码时容易混淆的地方

1. **②和③都可能叫“跳步”，但②移除网格节点，③只省模型预测。** ③④每个原始区间仍执行 Scheduler 更新。
2. **已有预测可以决定后续步骤。** ①②使用当前真实预测判断未来，不是白算一个将被跳过的当前预测。
3. **Feature、主干残差、最终预测是三种不同张量。** Feature 用来观察或作为主干输入；残差是 Z-H；最终预测是输出头之后给 Scheduler 的张量。
4. **③④的 NFE 只计完整主干计算。** ④复用时仍有输出头开销，入口 feature 也有开销，不能把它们当成完全免费。
5. **CFG 是批处理。** 一次 batched forward 可以包含两个分支；`denoiser_items` 与 NFE 不是同一计数。
6. **修改框架模板不会自动改写旧运行目录。** 旧策略副本与当前默认模板要分别看。
7. **Python 策略决定动作，LLM 改策略代码。** 两者的耗时与调用频率要分开理解。

相关记录字段包括 `scheduler_updates`、`feature_extractions`、`output_head_evaluations`、`feature_seconds`、`policy_seconds`、`denoiser_seconds`、`cache_reuse_seconds`、`agent_seconds`。

环境安装、烟雾推理及其专用依赖配置已按用户要求删除；现有运行环境、模型、资源和正式运行入口保留。

## 10. 代码检查入口与设计参考

使用现有 Python 环境，在 tmux 中运行 CPU 回归检查：

```bash
python -m unittest discover -s tests -v
```

核心测试位置：

- `tests/test_core.py`、`test_grid_skip.py`：①②控制流程。
- `tests/test_cache_methods.py`：③④真实小模型的全 keep 对齐、复用时主干不执行、缓存重置、CFG 与预测前检查。
- `tests/test_config.py`：配置展开、步数继承和 JSON 配置生成。
- `tests/test_sd3_scheduler.py`、`test_sdxl_grid.py`：官方 Scheduler 与模型适配。
- `tests/test_recovery.py`：策略更新、失败回退及 mailbox 行为。

这些工具用于理解和验证实现，不代表默认策略已经产生加速或保证质量。

两个最直接的设计参考：

- [AdaptiveDiffusion，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/00d1f03b87a401b1c7957e0cc785d0bc-Paper-Conference.pdf)：旧噪声预测复用，同时保留逐节点 latent 更新，对应③的缓存思路。
- [TeaCache 官方 FLUX 实现，CVPR 2025 方法](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py)：缓存主干输出减输入的残差，对应④的缓存思路。

本项目的六字段浅层 observation 和 SDXL 整主干缓存边界属于自己的实现选择，不应称为完整复现上述论文。研究重点是 Agent 如何发现有效的采样/复用策略，而不是把缓存复用本身作为新的发明。
