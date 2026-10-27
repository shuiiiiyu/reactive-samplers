"""Prepare three paired COCO pipeline checks for Trento."""
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers")
RUN = "coco2-pipeline-20261027"

out = ROOT / "sampler_runs" / RUN
out.mkdir(parents=True, exist_ok=False)
rows = (ROOT / "sampler_runs/coco10-grid-20261027/prompts.jsonl").read_text().splitlines()[:2]
(out / "prompts.jsonl").write_text("\n".join(rows) + "\n")
with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
    base = OmegaConf.to_container(compose(config_name="sd3_main"), resolve=True)
for method in ("grid_skip", "prediction_cache", "residual_cache"):
    config = dict(base, method=method, mode="explore", reference_steps=40, max_nfe=40,
                  expected_count=2, prompts=str(REMOTE / "sampler_runs" / RUN / "prompts.jsonl"),
                  output_dir=str(REMOTE / "sampler_runs" / RUN / method),
                  model_path=str(REMOTE / "models/sd3-medium"))
    config["agent"] = dict(base["agent"], schedule="every_step", checkpoints=[])
    config["metrics"] = {key: str(REMOTE / value) if key != "names" else value
                         for key, value in base["metrics"].items()}
    (out / (method + ".json")).write_text(json.dumps(config, indent=2) + "\n")
print(out)
