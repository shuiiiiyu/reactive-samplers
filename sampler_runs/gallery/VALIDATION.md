# Reactive Samplers 图片网页

已部署于 Trento，个人访问地址：<http://127.0.0.1:18765>（需 SSH 隧道）。

- SD3 / SDXL 官方生成接口；在网页填写 Prompt、Steps、CFG、Seed 和尺寸。
- `sampler_runs/` 内图片和已有生成记录自动显示，每 4 秒刷新。
- 最多选择 6 张同屏对比，可展开查看完整配置。
- 新生成记录包含实际 timestep/sigma 序列、NFE、耗时、显存、随机种子及模型环境版本。

已从网页真实提交并完成两种模型的生成，Slurm 作业 48049043、48049067 均为 COMPLETED，退出码 0:0；新图无需刷新即可出现。网页目前展示 6 张真实图片，已验证六图选择、同屏对比和配置弹窗。新增 3 项 HTTP / 数据兼容性 / 参数与访问边界测试通过。未使用 Agent。

以后打开：运行交付的 `open-reactive-gallery.ps1`，它会检查服务并重连两段 SSH 隧道。网页或电脑关闭后，已提交的 Slurm 作业会继续运行。

Trento 项目：`/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers`。

主要文件：`reactive_samplers/gallery.py`、`reactive_samplers/generation.py`、`reactive_samplers/web/index.html`、`scripts/run_trento_gallery_job.sh`。接口与运行说明在 `docs/gallery.md`；服务日志在 `sampler_runs/gallery/server.log`；网页生成任务记录在 `sampler_runs/web/<request_id>/`。集群私有启动配置放在忽略目录 `.venv-trento/`。

接口：`GET /api/models`、`GET /api/images`、`GET /api/jobs`、`POST /api/generate`。详情见一并交付的接口说明。

新代码和验证记录保存在 Trento，本机也留有源代码副本；本次没有推送 GitHub。已有 smoke test 和研究框架的图片只要继续保存到 `sampler_runs/`，无需改成手动上传流程。
