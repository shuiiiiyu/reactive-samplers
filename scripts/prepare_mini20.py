"""Write concrete paired SD3 configs for the 20-prompt main experiment."""

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "sampler_runs/mini20-20261026"
PROMPTS = ROOT / "sampler_runs/datasets/coco_hpd_20_v1"
BASE = {
    "model_path": str(ROOT / "models/sd3-medium"), "device": "cuda", "dtype": "bfloat16",
    "height": 1024, "width": 1024, "guidance_scale": 7.0,
    "max_sequence_length": 256, "reference_steps": 28, "max_nfe": 28,
    "expected_count": 10, "policy": None,
    "agent": {"backend": "mailbox", "command": [], "timeout_seconds": 600,
              "poll_seconds": 1, "schedule": "every_step", "checkpoints": []},
    "metrics": {"names": ["hps_v2_1", "clip_score"],
                "clip_model": str(ROOT / "metrics/clip/checkpoints/ViT-B-32.pt"),
                "hps_checkpoint": str(ROOT / "benchmarks/hps/checkpoints/hpsv2/HPS_v2.1_compressed.pt")},
}


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    baseline_policy = RUNS / "default_policy"
    baseline_policy.mkdir(exist_ok=True)
    (baseline_policy / "analysis.py").write_text(
        "FEATURE_LAYERS = []\nHISTORY_LIMIT = 0\ndef analyze(ctx, memory):\n    return {}\n")
    shutil.copyfile(ROOT / "reactive_samplers/prompts/updater.py", baseline_policy / "updater.py")
    for name, mode, split in (("baseline_explore", "default", "explore"),
                              ("baseline_holdout", "default", "holdout"),
                              ("agent_explore", "explore", "explore"),
                              ("frozen_holdout", "evaluate", "holdout")):
        config = {**BASE, "mode": mode, "prompts": str(PROMPTS / f"{split}.jsonl"),
                  "output_dir": str(RUNS / name),
                  "baseline_results": str(RUNS / "baseline_explore/results.json") if mode == "explore" else None,
                  "policy": str(RUNS / "agent_explore/frozen_policy") if mode == "evaluate"
                            else str(baseline_policy) if mode == "default" else None}
        (RUNS / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(RUNS)


if __name__ == "__main__":
    main()
