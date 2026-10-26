# Trento 官方 Pipeline Smoke Test

Slurm 作业：48036668，最终状态 COMPLETED，退出码 0:0，总耗时 6 分 27 秒。未启动 Agent，未进行策略探索或正式 Benchmark。

项目目录：`/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers`。

使用 NVIDIA H100，Python 3.11.5，独立安装的 PyTorch 2.5.1+cu121、Diffusers 0.35.1、Transformers 4.44.2。已确认导入的是新环境内的 PyTorch，而非系统模块里的 2.4 预发布版。所有依赖已锁定，安装使用校验过的离线 wheels。

两个模型均使用 Trento 上已有的官方权重、原采样器、CPU offload，固定 seed=42，分辨率 1024×1024。

| 模型 | 步数 / NFE | 推理时间 | 模型加载时间 | 状态 |
|---|---:|---:|---:|---|
| SD3-Medium | 28 / 28 | 8.35 秒 | 40.69 秒 | 图片生成成功 |
| SDXL Base 1.0 | 30 / 30 | 4.04 秒 | 50.86 秒 | 图片生成成功 |

推理时间为脚本记录的完整 Pipeline 调用耗时，不包含模型加载；使用回调检查 latent 有限、统计实际 denoiser 调用数，并检查 PNG 尺寸、像素非空及可读性。两张图片也已查看。以上耗时仅为本次环境健康检查，不能用于宣称方法加速或质量保持。

- [SD3 图片](images/sd3.png)
- [SDXL 图片](images/sdxl.png)
- [详细生成记录](images/results.json)
- [环境信息](environment.json)
- [11 项通过的单元测试](unit-tests.log)
- [GPU 信息](nvidia-smi.txt)

Trento 图片目录：`sampler_runs/trento_smoke/48036668/images/`。

可复用测试入口：`tests/run_trento_smoke.sh`，在加载私有配置要求的模块后，于 Slurm GPU 分配中执行。已经生成的提交脚本为 `sampler_runs/setup/trento/smoke.sbatch`，可在 Trento 项目根目录执行：

```bash
sbatch sampler_runs/setup/trento/smoke.sbatch
```

环境在计算节点的临时目录中构建，已保存为 `.venv-trento/runtime.tar.gz`（约 2.8 GiB），以后运行时由测试入口解压复用。归档 SHA256：`9c7feb5bfb838157861b482d7a12632a9053d3aace8987b95d65854169e3ddd3`。无需将 RunPod 虚拟环境复制到集群。依赖源文件和锁文件在 `tests/trento-requirements.in`、`tests/trento-requirements.lock`；离线 wheels 位于忽略目录 `.venv-trento-wheels/`。

当前检查覆盖官方 Pipeline 的模型加载、CUDA 推理和现有单元测试，不代表自适应 Pipeline 端到端等价性或研究实验已经完成。本次修改没有推送 GitHub。
