import argparse
import json
from pathlib import Path

from .agent import AgentSession
from .backends import CommandAgent, MailboxAgent
from .programs import Programs, freeze
from .report import Recorder, summarize, write_json
from .schedule import CheckSchedule
from .wrapper import sample
from .context import METHODS


def read_prompts(path, expected):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} prompt/seed rows; found {len(rows)}")
    seen = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in row["id"]):
            raise ValueError("Prompt ids must be unique filename-safe strings")
        if row["id"] in seen or not isinstance(row.get("prompt"), str) or not row["prompt"].strip() or type(row.get("seed")) is not int:
            raise ValueError("Each row needs a unique id, nonempty prompt and integer seed")
        seen.add(row["id"])
    return rows


def run_config(cfg):
    """Shared main-experiment entry. Hydra resolves required fields before this function."""
    mode = cfg["mode"]
    method = cfg.get("method", "adaptive_sigma")
    if method not in METHODS:
        raise ValueError(f"Unknown sampling method: {method}")
    cfg = {**cfg, "method": method}
    model = cfg.get("model", "sd3")
    if model not in {"sd3", "sdxl"} or (model == "sdxl" and method != "grid_skip"):
        raise ValueError("Supported models: SD3; SDXL with grid_skip")
    cfg["model"] = model
    if mode not in {"default", "explore", "evaluate"}:
        raise ValueError("mode must be default, explore or evaluate")
    if mode == "evaluate" and not cfg.get("policy"):
        raise ValueError("Evaluation requires a frozen policy directory")
    agent_backend = cfg["agent"]["backend"]
    if mode == "explore" and agent_backend not in {"command", "mailbox"}:
        raise ValueError("Exploration requires the command or mailbox Agent backend")
    if mode == "explore" and agent_backend == "command":
        command_agent = CommandAgent(cfg["agent"]["command"], cfg["agent"]["timeout_seconds"])
    elif mode == "explore":
        command_agent = MailboxAgent(cfg["agent"]["timeout_seconds"], cfg["agent"]["poll_seconds"])
    else:
        command_agent = None
    schedule = CheckSchedule(cfg["agent"]["schedule"], tuple(cfg["agent"]["checkpoints"]))
    if mode == "explore" and schedule.mode == "checkpoints" and not schedule.checkpoints:
        raise ValueError("Checkpoint exploration requires at least one Agent checkpoint")
    prompts = read_prompts(cfg["prompts"], cfg["expected_count"])
    if mode == "evaluate" and not (Path(cfg["policy"]) / "manifest.json").is_file():
        raise ValueError("Evaluation requires a frozen manifest, not a mutable workspace")
    # All model imports are lazy; help/freeze/core tests need no CUDA or checkpoint.
    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline
    from .evaluation import QualityMetrics
    from .models.sd3 import SD3Adapter, save_context
    from .models.sdxl import SDXLAdapter, verify_pipeline
    if diffusers.__version__ != "0.35.1":
        raise RuntimeError("The initial SD3 adapter targets diffusers==0.35.1; run parity before changing versions")
    root = Path(cfg["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", cfg)
    (root / "prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts), encoding="utf-8")
    write_json(root / "environment.json", {"torch": torch.__version__, "diffusers": diffusers.__version__})
    from .recovery import FailureLog, POLICY_ERRORS
    try:
        metrics = QualityMetrics(cfg["metrics"]["names"], cfg["metrics"].get("clip_model"),
                                 cfg["metrics"].get("hps_checkpoint"))
        source = cfg.get("policy") if mode in {"default", "evaluate"} else None
        if source is None and model == "sdxl":
            source = Path(__file__).parent / "prompts/sdxl_grid_skip"
        programs = Programs.create(root / "workspace", root / "programs", source, method=method)
        pipeline_class = StableDiffusion3Pipeline if model == "sd3" else StableDiffusionXLPipeline
        pipe = pipeline_class.from_pretrained(
            cfg["model_path"], torch_dtype=getattr(torch, cfg["dtype"]), local_files_only=True,
        ).to(cfg["device"])
        adapter_class = SD3Adapter if model == "sd3" else SDXLAdapter
        adapter = adapter_class(pipe, **{k: cfg[k] for k in ("guidance_scale", "height", "width", "max_sequence_length")})
        if model == "sdxl" and cfg.get("verify_pipeline", False):
            parity = verify_pipeline(pipe, adapter, prompts[0]["prompt"], prompts[0]["seed"], cfg["reference_steps"])
            write_json(root / "official_pipeline_parity.json", parity)
    except POLICY_ERRORS as exc:
        FailureLog(root).record(exc, stage="initialization", outcome="aborted")
        write_json(root / "status.json", {"status": "failed", "stage": "initialization", "error": str(exc)})
        raise
    agent = None
    if mode == "explore":
        agent = AgentSession(command_agent, root, save_context, method=method, model=model)
    results = []
    reference = []
    if mode == "explore" and cfg.get("baseline_results"):
        reference = json.loads(Path(cfg["baseline_results"]).read_text(encoding="utf-8"))["images"]
        write_json(root / "feedback.json", {"reference_images": reference, "completed_images": results})
    for row in prompts:
        image_dir = root / "images"
        image_dir.mkdir(exist_ok=True)
        recorder = Recorder(root / "trajectories" / row["id"], run_root=root)
        stage = "sampling"
        try:
            image, stats, history = sample(adapter, programs, row["prompt"], row["seed"], cfg["reference_steps"],
                                           cfg["max_nfe"], schedule, recorder, agent,
                                           policy_timeout_seconds=cfg.get("policy_timeout_seconds", 30), method=method)
            stage = "image_save"
            image_path = image_dir / (row["id"] + ".png")
            image.save(image_path)
            stage = "metrics"
            scores = metrics.score(image_path, row["prompt"])
        except POLICY_ERRORS as exc:
            if stage != "sampling":
                FailureLog(root).record(exc, stage=stage, version=programs.version, outcome="aborted")
            write_json(root / "status.json", {"status": "failed", "stage": stage,
                                               "sample_id": row["id"], "error": str(exc)})
            raise
        result = {**row, **stats, "scores": scores, "image": str(image_path)}
        results.append(result)
        if mode == "explore":
            # Full raw history is retained only in exploration; frozen inference keeps policy memory only.
            torch.save(history, recorder.root / "history.pt")
        write_json(root / "feedback.json", {"reference_images": reference, "completed_images": results})
        clean = [item for item in results if item["valid_for_benchmark"]]
        write_json(root / "results.json", {"images": results, "summary": summarize(results),
                   "clean_summary": summarize(clean) if clean else None,
                   "benchmark_status": "clean" if len(clean) == len(results) else "contains_failed_or_recovered_samples",
                   "quality_status": "measured" if cfg["metrics"]["names"] else "not_configured"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del history
    if mode == "explore":
        try:
            freeze(programs.versions / programs.version, root / "frozen_policy", method=method)
        except POLICY_ERRORS as exc:
            FailureLog(root).record(exc, stage="freeze", version=programs.version, outcome="aborted")
            write_json(root / "status.json", {"status": "failed", "stage": "freeze", "error": str(exc)})
            raise
    write_json(root / "status.json", {"status": "completed", "completed_images": len(results),
                                       "recovered_images": sum(row["recovered"] for row in results)})


def main():
    parser = argparse.ArgumentParser(description="Reactive Samplers framework utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("freeze", help="Export a complete immutable-by-hash strategy")
    export.add_argument("source")
    export.add_argument("target")
    export.add_argument("--method", choices=METHODS, default="adaptive_sigma")
    inspect = sub.add_parser("inspect", help="Run edited analysis against a paused trusted state snapshot")
    inspect.add_argument("state")
    inspect.add_argument("--workspace", required=True)
    inspect.add_argument("--output")
    report = sub.add_parser("report", help="Print existing recorded results; makes no model calls")
    report.add_argument("run_dir")
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.source, args.target, method=args.method))
    elif args.command == "report":
        print((Path(args.run_dir) / "results.json").read_text(encoding="utf-8"))
    else:
        import tempfile
        from .models.sd3 import load_context
        with tempfile.TemporaryDirectory() as temp:
            ctx = load_context(args.state)
            programs = Programs(args.workspace, Path(temp) / "versions", method=ctx.method)
            observation = programs.analysis.analyze(ctx, ctx.analysis_memory)
            if args.output:
                write_json(args.output, observation)
            print(json.dumps(observation, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
