"""Official local-weight generation used by the gallery's Slurm jobs; no Agent."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

MODELS = {
    "sd3": {"name": "SD3 Medium", "folder": "sd3-medium", "steps": 28, "guidance_scale": 7.0,
            "dtype": "bfloat16", "pipeline": "StableDiffusion3Pipeline", "sampler": "FlowMatchEulerDiscreteScheduler"},
    "sdxl": {"name": "SDXL Base 1.0", "folder": "sdxl-base-1.0", "steps": 30, "guidance_scale": 5.0,
             "dtype": "float16", "pipeline": "StableDiffusionXLPipeline", "sampler": "EulerDiscreteScheduler"},
}


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalize_request(value):
    if not isinstance(value, dict):
        raise ValueError("请求必须是 JSON 对象")
    allowed = {"model", "prompt", "negative_prompt", "seed", "steps", "guidance_scale", "width", "height"}
    if set(value) - allowed:
        raise ValueError("存在不支持的生成参数")
    model = value.get("model")
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError("模型必须是 sd3 或 sdxl")
    result = {"model": model, "negative_prompt": "", "seed": 42, "width": 1024, "height": 1024,
              "steps": MODELS[model]["steps"], "guidance_scale": MODELS[model]["guidance_scale"], **value}
    for key in ("prompt", "negative_prompt"):
        if not isinstance(result.get(key), str) or len(result[key]) > 4096:
            raise ValueError(f"{key} 必须是最多 4096 字的文本")
    if not result["prompt"].strip():
        raise ValueError("Prompt 不能为空")
    for key, low, high in (("seed", 0, 2**32 - 1), ("steps", 1, 100), ("width", 256, 1536), ("height", 256, 1536)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} 必须是 {low}–{high} 之间的整数")
    if result["width"] % 64 or result["height"] % 64:
        raise ValueError("宽和高必须是 64 的倍数")
    cfg = result["guidance_scale"]
    if type(cfg) not in (int, float) or not 0 <= cfg <= 20:
        raise ValueError("CFG 必须在 0–20 之间")
    return result


def generate(request_path):
    request_path = Path(request_path).resolve()
    run = request_path.parent
    status = run / "status.json"
    write_atomic(status, {"status": "running", "stage": "loading"})
    try:
        import numpy as np
        import torch
        import diffusers
        import transformers
        config = normalize_request(json.loads(request_path.read_text(encoding="utf-8")))
        if not torch.cuda.is_available():
            raise RuntimeError("Generation requires a Slurm CUDA allocation")
        spec = MODELS[config["model"]]
        project = Path(__file__).resolve().parent.parent
        started = time.perf_counter()
        pipe = getattr(diffusers, spec["pipeline"]).from_pretrained(
            project / "models" / spec["folder"], torch_dtype=getattr(torch, spec["dtype"]),
            local_files_only=True, use_safetensors=True,
        )
        if type(pipe.scheduler).__name__ != spec["sampler"]:
            raise RuntimeError("Checkpoint scheduler differs from the supported official sampler")
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        load_seconds = time.perf_counter() - started
        counts = {"nfe": 0}
        denoiser = pipe.transformer if config["model"] == "sd3" else pipe.unet
        def forward_count(*_):
            counts["nfe"] += 1
        hook = denoiser.register_forward_hook(forward_count)
        def progress(pipeline, step, timestep, kwargs):
            if not torch.isfinite(kwargs["latents"]).all().item():
                raise FloatingPointError("Nonfinite latent")
            write_atomic(status, {"status": "running", "stage": "sampling", "step": step + 1, "total": config["steps"]})
            return kwargs
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = pipe(
            prompt=config["prompt"], negative_prompt=config["negative_prompt"],
            num_inference_steps=config["steps"], guidance_scale=config["guidance_scale"],
            width=config["width"], height=config["height"],
            generator=torch.Generator(device="cuda").manual_seed(config["seed"]),
            callback_on_step_end=progress,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        hook.remove()
        image = output.images[0]
        pixels = np.asarray(image)
        if image.size != (config["width"], config["height"]) or pixels.std() < 1:
            raise RuntimeError("Generated image is empty or has the wrong dimensions")
        image_path = run / "image.png"
        image.save(run / "image.partial.png")
        (run / "image.partial.png").replace(image_path)
        entry = {**config, "status": "complete", "image": str(image_path), **counts,
                 "sampler": type(pipe.scheduler).__name__, "scheduler_config": dict(pipe.scheduler.config),
                 "timesteps": pipe.scheduler.timesteps.detach().cpu().tolist(),
                 "sigmas": pipe.scheduler.sigmas.detach().cpu().tolist(),
                 "dtype": spec["dtype"], "cpu_offload": True, "load_seconds": load_seconds,
                 "inference_seconds": elapsed, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "mode": "default", "agent_used": False}
        write_atomic(run / "results.json", {"prompt": config["prompt"], "seed": config["seed"],
                     "created_at": datetime.now(timezone.utc).isoformat(),
                     "environment": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                                     "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
                     "images": {config["model"]: entry}})
        write_atomic(status, {"status": "complete"})
    except Exception as exc:
        write_atomic(status, {"status": "failed", "error": str(exc)[:2000]})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    generate(parser.parse_args().request)
