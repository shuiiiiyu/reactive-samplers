import argparse
import json
from pathlib import Path

from .agent import AgentSession
from .backends import CommandAgent
from .programs import Programs, freeze
from .report import Recorder, summarize, write_json
from .schedule import CheckSchedule
from .wrapper import sample


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
    if mode not in {"default", "explore", "evaluate"}:
        raise ValueError("mode must be default, explore or evaluate")
    if mode == "evaluate" and not cfg.get("policy"):
        raise ValueError("Evaluation requires a frozen policy directory")
    if mode == "explore" and cfg["agent"]["backend"] != "command":
        raise ValueError("Exploration requires a real configured agent bridge; keep is only for smoke checks")
    command_agent = CommandAgent(cfg["agent"]["command"]) if mode == "explore" else None
    schedule = CheckSchedule(cfg["agent"]["schedule"], tuple(cfg["agent"]["checkpoints"]))
    if mode == "explore" and schedule.mode == "checkpoints" and not schedule.checkpoints:
        raise ValueError("Checkpoint exploration requires at least one Agent checkpoint")
    prompts = read_prompts(cfg["prompts"], cfg["expected_count"])
    if mode == "evaluate" and not (Path(cfg["policy"]) / "manifest.json").is_file():
        raise ValueError("Evaluation requires a frozen manifest, not a mutable workspace")
    # All model imports are lazy; help/freeze/core tests need no CUDA or checkpoint.
    import torch
    import diffusers
    from diffusers import StableDiffusion3Pipeline
    from .evaluation import QualityMetrics
    from .models.sd3 import SD3Adapter, save_context
    if diffusers.__version__ != "0.35.1":
        raise RuntimeError("The initial SD3 adapter targets diffusers==0.35.1; run parity before changing versions")
    metrics = QualityMetrics(cfg["metrics"]["names"], cfg["metrics"].get("clip_model"))
    root = Path(cfg["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", cfg)
    (root / "prompts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts), encoding="utf-8")
    write_json(root / "environment.json", {"torch": torch.__version__, "diffusers": diffusers.__version__})
    source = cfg.get("policy") if mode == "evaluate" else None
    programs = Programs.create(root / "workspace", root / "programs", source)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg["model_path"], torch_dtype=getattr(torch, cfg["dtype"]), local_files_only=True,
    ).to(cfg["device"])
    adapter = SD3Adapter(pipe, **{k: cfg[k] for k in ("guidance_scale", "height", "width", "max_sequence_length")})
    agent = None
    if mode == "explore":
        agent = AgentSession(command_agent, root, save_context)
    results = []
    for row in prompts:
        image_dir = root / "images"
        image_dir.mkdir(exist_ok=True)
        recorder = Recorder(root / "trajectories" / row["id"])
        image, stats, history = sample(adapter, programs, row["prompt"], row["seed"], cfg["reference_steps"],
                                       cfg["max_nfe"], schedule, recorder, agent)
        image_path = image_dir / (row["id"] + ".png")
        image.save(image_path)
        scores = metrics.score(image_path, row["prompt"])
        result = {**row, **stats, "scores": scores, "image": str(image_path)}
        results.append(result)
        if mode == "explore":
            # Full raw history is retained only in exploration; frozen inference keeps policy memory only.
            torch.save(history, recorder.root / "history.pt")
        write_json(root / "feedback.json", {"completed_images": results})
        write_json(root / "results.json", {"images": results, "summary": summarize(results),
                                             "quality_status": "measured" if cfg["metrics"]["names"] else "not_configured"})
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del history
    if mode == "explore":
        freeze(programs.versions / programs.version, root / "frozen_policy")


def main():
    parser = argparse.ArgumentParser(description="Reactive Samplers framework utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("freeze", help="Export a complete immutable-by-hash strategy")
    export.add_argument("source")
    export.add_argument("target")
    inspect = sub.add_parser("inspect", help="Run edited analysis against a paused trusted state snapshot")
    inspect.add_argument("state")
    inspect.add_argument("--workspace", required=True)
    inspect.add_argument("--output")
    report = sub.add_parser("report", help="Print existing recorded results; makes no model calls")
    report.add_argument("run_dir")
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.source, args.target))
    elif args.command == "report":
        print((Path(args.run_dir) / "results.json").read_text(encoding="utf-8"))
    else:
        import tempfile
        from .models.sd3 import load_context
        with tempfile.TemporaryDirectory() as temp:
            programs = Programs(args.workspace, Path(temp) / "versions")
            ctx = load_context(args.state)
            observation = programs.analysis.analyze(ctx, ctx.analysis_memory)
            if args.output:
                write_json(args.output, observation)
            print(json.dumps(observation, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
