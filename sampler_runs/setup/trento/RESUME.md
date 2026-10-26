# Reactive Samplers：Trento 环境与 Smoke Test 交接

## 05:00 续跑后的完成更新

本任务已完成。Trento 上使用独立 PyTorch 2.5.1+cu121 环境，通过 Slurm 作业 48036668，在 NVIDIA H100 上完成 SD3 28 步和 SDXL 30 步各一张 1024×1024 官方 Pipeline 图片生成。11 项单元测试通过；作业 COMPLETED、退出码 0:0、总耗时 6 分 27 秒。没有 Agent 调用，没有正式 Benchmark。

环境归档已保存至项目 `.venv-trento/runtime.tar.gz`；入口为 `tests/run_trento_smoke.sh`，提交脚本 `sampler_runs/setup/trento/smoke.sbatch`。图片和记录在 `sampler_runs/trento_smoke/48036668/`，详细报告 `REPORT.md`；本机副本在本交接文件同级的 `trento-smoke-48036668/README.md`。

SD3 Pipeline 调用耗时 8.35 秒，SDXL 4.04 秒，不包含模型加载；不能用于宣称自适应方法加速。新测试脚本和依赖锁文件保存在 Trento，尚未推送 GitHub。安装缓存放在隐藏的 `.venv*` 忽略目录中。下方为暂停前的历史记录，已被本节更新。

保存时间：2026-10-26 02:54 左右，北京时间。

用户要求：现在暂停，以节省 Codex 额度；今天 05:00 继续建立 Trento 环境并运行最小 smoke test，先不用 Agent。已获授权传输文件、创建环境和提交必要的 Slurm GPU 作业。不要重新询问这些授权，不要自动推送 GitHub。

## 已完成

- 完整项目已由 RunPod 迁到 Trento，7,448 个文件，49,494,833,182 字节。全量 checksum 对齐；检查 Git 状态引起的三个子仓库 index 刷新也已重新对齐并单独校验。代码、模型、数据和运行记录均在 Trento。
- Trento 项目：`/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers`。
- RunPod 原项目：`/workspace/home/shencanyu/deepresearch/workspace/reactive-samplers`。每日 06:00 可能清理，因此不要依赖它长期保存项目。
- GitHub：`https://github.com/vibe-hust/reactive-samplers`，main 提交 `acb02b839190608dc9cf6bcc0166125fe9e166af`；权重和数据未上传 GitHub。
- Trento 已配置模块提供 Python 3.11.5、uv 0.10.11、PyTorch 2.4.0a0+gite3b9b71。不能将此预发布版假称为已验证的完整推理环境。
- 根目录 pyproject.toml 的 sd3 可选依赖是 torch>=2.5；这是此前工程约束，不是已核实的 SD3 官方最低要求。用户追问时已经明确纠正过这个区别。

## 正在进行但尚未完成

- 在 RunPod 准备 Python 3.11 的独立、固定版本推理环境所需 wheels，计划在 Trento 安装官方 torch 2.5.1+cu121，diffusers 0.35.1，transformers 4.44.2，accelerate 1.10.1 等。
- RunPod 的 uv 已安装 Python 3.11.13。
- 新增 `tests/trento-requirements.in` 和 `tests/trento-requirements.lock`（已成功解析锁定全部依赖）。
- staging 脚本及日志在 `sampler_runs/setup/trento/stage.sh`、`stage.log`、`stage.status`。
- 用户要求暂停后，已关闭唯一的 `reactive-trento-env-stage` tmux 会话。暂停时 pip 正下载 torch wheel；wheels 目录还没有完整 wheel。不要误报下载完成或环境已经建好。
- **没有提交任何 Trento GPU 作业，Trento 的新环境尚未建立，smoke test 尚未运行。**

## 恢复入口

本地控制目录：`C:\Users\24778\Documents\Codex\2026-10-25\chat`。

本地辅助文件：

- `work/trento_transfer_control.py`：安全读取 `F:\科研\trento-config-files\config_trento.txt`，不打印私有配置。configuration() 返回已填写的键值。
- `work/trento_command.py`：`python work/trento_command.py <本地命令文件> --modules` 经 RunPod SSH 到 Trento，先加载配置里的 TRENTO_RUNTIME_MODULE_SETUP。
- `work/prepare_trento_smoke.py`：重新写入 staging 脚本并在 RunPod 的独立 tmux 会话启动依赖准备。脚本可利用已下载的 Python / pip 缓存，随后产生 wheels 和 SHA256 清单。
- `work/inspect_trento_qos.py`：查询配置中的验证 QoS，已确认 acc_debug 的 MaxWall=02:00:00。

SSH 路由：本机 `ssh -o BatchMode=yes vibelab-runpod ...` → RunPod 的 `ssh -o BatchMode=yes trento ...`。这是已经成功验证的路由，Trento 用户 tren811487。不要复制密钥或 HF token，不打印凭据。网络命令目前需 exec_command 的 require_escalated，已多次获自动审核许可。

私有配置包含 TRENTO_SSH、TRENTO_DEEPRESEARCH_ROOT、TRENTO_ACCOUNT、TRENTO_PARTITION、TRENTO_VALIDATION_QOS、TRENTO_CPUS_PER_GPU、TRENTO_RUNTIME_MODULE_SETUP 等。读取真实值生成作业，不能猜测账户。当前登记根存在：`/gpfs/scratch/ehpc1056/TengLong/deepresearch`。

## 下一步

1. 检查 staging 会话确实已停，再运行 `python work/prepare_trento_smoke.py` 恢复 wheels 准备。长任务必须由 RunPod tmux 或 Slurm 管理，不能依赖本机 SSH 长连接。
2. 只同步新的 requirements、测试脚本和 runtime wheels；不要重新传 49.5 GB 项目。不要复制 RunPod 的 venv 或 resolver-env。
3. Trento 使用 uv 创建自己的独立环境。配置模块会通过 PYTHONPATH 注入旧 PyTorch；独立安装 torch 时应清理该注入，实际检查导入路径和版本，防止安装新版却导入旧版。如 CUDA 模块库覆盖 wheel 附带库，需要让 wheel 内 NVIDIA 库路径优先，并验证实际 CUDA 推理。
4. 遵循 Trento 已核实的调度规则：单 GPU 至少 20 CPU 核；不传 --mem / --mem-per-cpu；使用配置的 validation QoS；GPU 推理必须通过 Slurm 计算节点。优先在 node-local 临时目录构建环境并保存 Trento 构建的紧凑环境归档，便于后续恢复。
5. 复用 `tests/smoke_generate.py --model both --output-dir sampler_runs/<新的唯一run目录>`。它使用官方本地权重：SD3 28 步、SDXL 30 步，各生成一张 1024×1024 图片，记录 NFE、耗时、显存、输出路径并验证 latent finite / 图片非空。没有 Agent。原 `tests/run_inference_smoke.sh` 专用于 RunPod，不要直接拿来作为 Trento 环境启动器。
6. 等作业真正结束，核对 JSON、图片及退出码，再报告。smoke 成功仅说明权重和推理环境可用，不能说明自适应方法正确、质量保持或加速。

## 已解决的准备问题

- uv 默认多 index 优先级让 PyTorch index 的旧 setuptools 遮挡 PyPI。当前 staging 对官方 PyPI + 官方 PyTorch 两个可信 index 使用 `--index-strategy unsafe-best-match`；全部包版本已经锁定。
- Hydra 的 antlr4-python3-runtime==4.9.3 没有预编译 wheel，因此不能 pip download --only-binary=:all:。当前脚本已改用 `pip wheel`，用 Python 3.11 将该小型纯 Python 源码包构建成 wheel。
- staging 的 resolver-env 和 wheels 目前临时位于 RunPod 的 sampler_runs/setup/trento；不要把 resolver-env 复制到 Trento，也不要提交这些安装缓存到 GitHub。可在完成后只保留必要的锁文件、脚本和运行记录，避免污染已按 OPAL 整理的布局。

## 项目边界

读取项目 AGENTS.md。项目按 OPAL 组织；不要加入 SUE 的 scale_up_outputs 或改造研究框架。使用 DeepResearch 的 Trento 连接、模块和调度要求即可。所有实验日志和图片放 sampler_runs。现在只做环境和官方 Pipeline smoke，不做 Agent、策略探索、正式 Benchmark 或额外研究。

当地机器没有额度或未开机时自动化可能无法执行；不能声称定时任务必定完成，也不要自行变换模型或购买额度。
