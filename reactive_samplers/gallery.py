"""Private, dependency-free image gallery and Slurm generation API."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import quote, unquote, urlsplit
import uuid

from .generation import MODELS, normalize_request, write_atomic


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def contained_image(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Invalid image path")
    return path


def experiment_info(relative, runs):
    parts = Path(relative).parts
    name = parts[0]
    identity = "/".join(parts[:2]) if name == "trento_smoke" and len(parts) > 2 else name
    if name in {"smoke_generation", "smoke_generation_sd3"}:
        identity = "smoke_generation"
    info = read_json(runs / identity / "experiment.json")
    title = info.get("title") if isinstance(info, dict) else None
    if not title:
        label = ("COCO 十样本跳步对比实验" if name.startswith("coco10-grid") else
                 "小规模采样对比实验" if name.startswith("mini20") else
                 "Agent 接入验证" if name.startswith("agent-cli-smoke") else
                 "模型推理验证" if "smoke" in name else
                 "手动生成实验" if name == "web" else "采样实验")
        date = re.search(r"20\d{6}", name)
        suffix = identity
        if date:
            stamp = date[0]
            revision = name.split(stamp)[1].strip("-")
            suffix = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}" + (" · " + revision if revision else "")
        title = label + (" · " + suffix if name != "web" else "")
        if name in {"smoke_generation", "smoke_generation_sd3"}:
            title = "RunPod 模型推理验证"
        elif name == "trento_smoke":
            title = "Trento 模型推理验证 · 作业 " + parts[1]
    return {"id": identity, "title": title}


class Gallery:
    def __init__(self, project, template=None):
        self.project = Path(project).resolve()
        self.runs = self.project / "sampler_runs"
        self.template = Path(template).resolve() if template else None
        self.lock = threading.Lock()
        self.checked_at = 0

    def images(self):
        metadata = {}
        # Scan manifests directly; no database or separate upload step is needed.
        for report in self.runs.rglob("results.json"):
            data = read_json(report)
            if not isinstance(data, dict):
                continue
            entries = data.get("images", {})
            if not isinstance(entries, (dict, list)):
                continue
            config = read_json(report.parent / "config.json")
            if not isinstance(config, dict):
                config = {}
            common = {k: data[k] for k in ("prompt", "seed", "created_at", "environment") if k in data}
            for key, entry in (entries.items() if isinstance(entries, dict) else ((None, x) for x in entries)):
                if not isinstance(entry, dict) or entry.get("status", "complete") != "complete":
                    continue
                filename = str(entry.get("image", ""))
                if not filename:
                    continue
                candidates = [Path(filename), report.parent / filename, report.parent / Path(filename).name,
                              report.parent / "images" / Path(filename).name]
                for candidate in candidates:
                    candidate = candidate.resolve()
                    if candidate.is_relative_to(self.runs) and candidate.is_file():
                        public_config = {k: config[k] for k in ("model_path", "mode", "method", "dtype", "guidance_scale", "height", "width", "reference_steps", "max_nfe", "policy", "max_sequence_length") if k in config}
                        if isinstance(config.get("agent"), dict):
                            public_config["agent_schedule"] = {k: config["agent"][k] for k in ("schedule", "checkpoints") if k in config["agent"]}
                        model_path = str(config.get("model_path", "")).lower()
                        inferred_model = "sdxl" if "sdxl" in model_path else "sd3" if "sd3" in model_path else None
                        metadata[candidate] = {**public_config, **common, **entry, "model": entry.get("model", key or inferred_model)}
                        records = []
                        for record_path in sorted((report.parent / "agent_decisions").glob("*.json")):
                            record = read_json(record_path)
                            if record.get("sample_id") == entry.get("id") or any(
                                    item.get("sample_id") == entry.get("id") for item in record.get("executions", [])):
                                records.append(record)
                        metadata[candidate]["agent_decisions"] = records
                        trace = (report.parent / "trajectories" / str(entry.get("id", "")) / "events.jsonl").resolve()
                        if trace.is_relative_to(report.parent) and trace.is_file():
                            steps = []
                            for line in trace.read_text(encoding="utf-8").splitlines():
                                try:
                                    event = json.loads(line)
                                except ValueError:
                                    continue
                                if event.get("event") == "step":
                                    steps.append({k: event[k] for k in ("step", "sigma", "next_sigma", "timestep", "next_timestep", "version", "grid_decisions", "skipped_node", "scheduler_updated", "reuse_cache", "backbone_evaluated", "forced", "agent_decision", "agent_outcome", "agent_output") if k in event})
                            metadata[candidate]["sampling_trajectory"] = steps
                        break
        rows = []
        for path in self.runs.rglob("*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or ".partial." in path.name:
                continue
            try:
                path = contained_image(self.runs, str(path.relative_to(self.runs)))
                stat = path.stat()
            except (OSError, ValueError):
                continue
            config = metadata[path] if path in metadata else read_json(path.with_suffix(".json"))
            if not isinstance(config, dict):
                config = {}
            relative = path.relative_to(self.runs).as_posix()
            rows.append({"id": relative, "url": "/images/" + quote(relative, safe=""),
                         "modified": stat.st_mtime, "run": relative.rsplit("/", 1)[0], "config": config,
                         "experiment": experiment_info(relative, self.runs)})
        return sorted(rows, key=lambda x: x["modified"], reverse=True)

    def jobs(self):
        rows = []
        for request in (self.runs / "web").glob("*/request.json"):
            run = request.parent
            info = read_json(run / "submission.json")
            state = read_json(run / "status.json")
            rows.append({"id": run.name, "config": read_json(request), **info, **state})
        rows.sort(key=lambda x: x.get("submitted_at", 0), reverse=True)
        with self.lock:
            if time.monotonic() - self.checked_at > 15:
                pending = [r for r in rows if r.get("slurm_job_id") and r.get("status") not in {"complete", "failed"}]
                if pending:
                    ids = ",".join(str(r["slurm_job_id"]) for r in pending)
                    try:
                        result = subprocess.run(["sacct", "-X", "-nP", "-j", ids, "--format=JobID,State,ExitCode"], capture_output=True, text=True, timeout=5)
                        states = {parts[0]: parts[1] for line in result.stdout.splitlines() if len(parts := line.split("|")) >= 3}
                        for row in pending:
                            state = states.get(str(row["slurm_job_id"]), "")
                            if state and not state.startswith(("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")):
                                run = self.runs / "web" / row["id"]
                                latest = read_json(run / "status.json")
                                if latest.get("status") not in {"complete", "failed"}:
                                    error = (run / "slurm.err").read_text(encoding="utf-8", errors="replace")[-1500:] if (run / "slurm.err").exists() else state
                                    row.update(status="failed", error=error or state)
                                    write_atomic(run / "status.json", {"status": "failed", "error": row["error"]})
                    except (OSError, subprocess.TimeoutExpired):
                        pass  # Keep the last recorded state if accounting is temporarily unavailable.
                self.checked_at = time.monotonic()
        return rows[:20]

    def submit(self, value):
        config = normalize_request(value)
        if self.template is None or not self.template.is_file():
            raise RuntimeError("生成接口尚未配置 Slurm 提交脚本")
        run = self.runs / "web" / uuid.uuid4().hex
        run.mkdir(parents=True)
        write_atomic(run / "request.json", config)
        write_atomic(run / "status.json", {"status": "queued"})
        try:
            result = subprocess.run(["sbatch", "--parsable", "--output", str(run / "slurm.out"),
                                     "--error", str(run / "slurm.err"), str(self.template), str(run / "request.json")],
                                    capture_output=True, text=True, timeout=20, check=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError("Slurm 未返回合法的作业编号")
            write_atomic(run / "submission.json", {"slurm_job_id": job_id, "submitted_at": time.time()})
            return {"id": run.name, "slurm_job_id": job_id, "status": "queued"}
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            error = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            if "QOSMaxSubmitJobPerUserLimit" in error:
                error = "当前 GPU 队列的提交额度已满，请稍后重试。"
            write_atomic(run / "status.json", {"status": "failed", "error": error})
            raise RuntimeError(error) from exc


def handler(gallery):
    class Handler(BaseHTTPRequestHandler):
        def json_response(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/models":
                return self.json_response(200, {"models": MODELS, "can_generate": bool(gallery.template)})
            if path == "/api/images":
                return self.json_response(200, {"images": gallery.images()})
            if path == "/api/jobs":
                return self.json_response(200, {"jobs": gallery.jobs()})
            try:
                if path == "/":
                    file = Path(__file__).with_name("web") / "index.html"
                elif path.startswith("/images/"):
                    file = contained_image(gallery.runs, unquote(path[len("/images/"):]))
                else:
                    return self.json_response(404, {"error": "Not found"})
                body = file.read_bytes()
            except (OSError, ValueError):
                return self.json_response(404, {"error": "Not found"})
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/generate":
                return self.json_response(404, {"error": "Not found"})
            origin = self.headers.get("Origin")
            if self.headers.get("X-Requested-With") != "reactive-samplers" or (origin and origin != "http://" + self.headers.get("Host", "")):
                return self.json_response(403, {"error": "请从本网页或受信任的本地 API 客户端提交"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536 or self.headers.get_content_type() != "application/json":
                    raise ValueError("需要不超过 64KB 的 JSON 请求")
                value = json.loads(self.rfile.read(length))
                self.json_response(202, gallery.submit(value))
            except (ValueError, UnicodeError) as exc:
                self.json_response(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.json_response(503, {"error": str(exc)})

        def log_message(self, format, *args):
            if not args or str(args[1] if len(args) > 1 else "") != "200":
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--slurm-template", type=Path)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(Gallery(args.project, args.slurm_template)))
    print(f"Reactive Samplers: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
