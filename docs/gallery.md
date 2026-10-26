# 图片网页

个人实验网页：浏览生成结果、查看配置、选择最多 6 张图片对比，直接提交 SD3 / SDXL 的官方默认采样任务。没有 Agent 调用。

网页直接读取 `sampler_runs/`，每 4 秒刷新。不需要手动上传图片，也没有额外数据库。原有 smoke test 和主实验的 `results.json`、`config.json` 都可读取；未记录的参数不会补成猜测值。主实验已有的逐步 sigma 和策略版本记录也会显示在配置中。

## 使用

服务仅监听 Trento 登录节点的 `127.0.0.1:8765`。访问链路是本机 SSH → 现有 RunPod → Trento。两段转发都只绑定 loopback，不开放公网端口。

已配置 RunPod 内层转发后，本机执行：

```bash
ssh -N -o ServerAliveInterval=30 -L 127.0.0.1:18765:127.0.0.1:18766 vibelab-runpod
```

打开 `http://127.0.0.1:18765`。重连助手在本次交付的 `open-reactive-gallery.ps1`，它会检查并重建 SSH 隧道。关闭浏览器或本机不会取消已经提交的 Slurm 作业；再次打开网页会读到已保存的结果。

Trento 服务在 tmux 会话 `reactive-gallery` 中运行。加载已配置模块及 `ncurses/6.4`、`tmux/3.3a` 后，可用以下命令重新启动；不要重复启动已经存在的同名会话：

```bash
tmux new-session -d -s reactive-gallery 'bash .venv-trento/start-gallery.sh'
```

从项目根目录执行。启动脚本和集群账户配置保存在忽略目录 `.venv-trento/`，没有写入公共网页。日志位于 `sampler_runs/gallery/server.log`。

## 接口

| 接口 | 用途 |
|---|---|
| `GET /api/models` | 模型标识、默认参数及接口是否已配置 |
| `GET /api/images` | 全部图片、配置和图片 URL |
| `GET /api/jobs` | 最近的生成请求及运行状态 |
| `POST /api/generate` | 校验参数后提交 Slurm GPU 作业 |

例如提交一张 SD3 图片：

```bash
curl http://127.0.0.1:18765/api/generate \
  -H 'Content-Type: application/json' \
  -H 'X-Requested-With: reactive-samplers' \
  -d '{"model":"sd3","prompt":"A red panda in a sunlit park","seed":42,"steps":28,"guidance_scale":7,"width":1024,"height":1024}'
```

支持 `sd3` 和 `sdxl`，可传 `negative_prompt`（默认空文本）。Steps 1–100，CFG 0–20，宽高 256–1536 且为 64 的倍数，Seed 为 32 位非负整数。请求文本写进 JSON 文件，不会拼接成 shell 命令。

请求保存在 `sampler_runs/web/<request_id>/request.json`。GPU 作业复用 Trento 已构建的环境归档，使用本地模型权重和官方采样公式，生成 `image.png`、`results.json` 和 `status.json`。配置记录包含实际 timestep/sigma、NFE、耗时、显存和环境版本。图片完成写入后会自动出现在网页。

网页控制面使用 Python 标准库，不需要安装 Node、前端构建工具或数据库。新增模型生成入口是 `reactive_samplers/generation.py`，通过 Slurm 调用；登录节点不加载生成模型。研究框架仍使用原有入口，结果保存在 `sampler_runs` 时也会自动显示。

## 检查

```bash
python -m unittest discover -s tests -p test_gallery.py -v
```

检查历史路径迁移、配置读取、参数校验、HTTP 请求、越界文件访问及跨来源提交限制。真实 GPU 生成联调的记录另存于 `sampler_runs/gallery/`。
